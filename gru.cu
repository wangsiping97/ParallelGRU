#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);

void mat_multiplication_kernel(float* a, float* b, float* c, int c_width, int c_height, int a_width) {
    for (int i = 0; i < c_height; ++i) {
        for (int j = 0; j < c_width; ++j) {
            for (int k = 0; k < a_width; ++k) {
                c[i * c_width + j] += a[i * a_width + k] * b[k * c_width + j];
            }
        }
    }
}

__global__ void 
mat_add_kernel(float* a, float* b, float* res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = a[index] + b[index];
}

__global__ void 
mat_add_b_kernel(float* a, float* b, float* res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = a[index] + b[index % width];
}

__global__ void 
mat_one_sub_kernel(float* a, float* res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = 1 - a[index];
}

__global__ void 
mat_hadamard_kernel(float*a, float* b, float* res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = a[index] * b[index];
}

__global__ void 
mat_sigmoid_kernel(float* a, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        a[index] = 1 / (1 + exp(-1 * a[index]));
}

__global__ void 
mat_tanh_kernel(float* a, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        a[index] = tanh(a[index]);
}

// x_t: width: 28, height: batch_size
// old_h_t: width: hidden_unit, height: batch_size
// new_h_t: width: hidden_unit, height: batch_size
// w_z, w_r, w_h: width: hidden_unit, height: 28
// u_z, u_r, u_h: width: hidden_unit, height: hidden_unit
// b_z, b_r, b_h: width: hidden_unit, height: 1
void gru_forward_kernel(int batch_size, int x_width, int hidden_unit,
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h) {
    
    // remove after paralleled matrix multiplication
    float* host_x_t = (float*)malloc(batch_size * x_width * sizeof(float));
    cudaMemcpy(host_x_t, x_t, batch_size * x_width * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_w_z = (float*)malloc(hidden_unit * x_width * sizeof(float));
    cudaMemcpy(host_w_z, w_z, hidden_unit * x_width * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_old_h_t = (float*)malloc(batch_size * hidden_unit * sizeof(float));
    cudaMemcpy(host_old_h_t, old_h_t, batch_size * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_u_z = (float*)malloc(hidden_unit * hidden_unit * sizeof(float));
    cudaMemcpy(host_u_z, u_z, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_w_r = (float*)malloc(hidden_unit * x_width * sizeof(float));
    cudaMemcpy(host_w_r, w_r, hidden_unit * x_width * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_u_r = (float*)malloc(hidden_unit * hidden_unit * sizeof(float));
    cudaMemcpy(host_u_r, u_r, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_w_h = (float*)malloc(hidden_unit * x_width * sizeof(float));
    cudaMemcpy(host_w_h, w_h, hidden_unit * x_width * sizeof(float), cudaMemcpyDeviceToHost);
    float* host_u_h = (float*)malloc(hidden_unit * hidden_unit * sizeof(float));
    cudaMemcpy(host_u_h, u_h, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);
    // end remove
    
    const int threadsPerBlock = 512;
    int blocks = (hidden_unit * batch_size + threadsPerBlock - 1) / threadsPerBlock;

    float* tmp1 = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    float* tmp2 = (float*)malloc(hidden_unit * batch_size * sizeof(float));

    float* device_tmp1;
    float* device_tmp2;
    cudaMalloc((void **)&device_tmp1, hidden_unit * batch_size * sizeof(float));
    cudaMalloc((void **)&device_tmp1, hidden_unit * batch_size * sizeof(float));

    // z_t = sigmoid(x_t * w_z + old_h_t * u_z + b_z)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_x_t, host_w_z, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_old_h_t, host_u_z, tmp2, hidden_unit, batch_size, hidden_unit);
    float* z_t;
    cudaMalloc((void **)&z_t, hidden_unit * batch_size * sizeof(float));

    // copy to device
    cudaMemcpy(device_tmp1, tmp1, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_tmp2, tmp2, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);

    mat_add_kernel<<<blocks, threadsPerBlock>>>(device_tmp1, device_tmp2, z_t, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(z_t, b_z, z_t, hidden_unit, batch_size);
    mat_sigmoid_kernel<<<blocks, threadsPerBlock>>>(z_t, hidden_unit, batch_size);

    // r_t = sigmoid(x_t * w_r + old_h_t * u_r + b_r)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_x_t, host_w_r, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_old_h_t, host_u_r, tmp2, hidden_unit, batch_size, hidden_unit);
    float* r_t;
    cudaMalloc((void **)&r_t, hidden_unit * batch_size * sizeof(float));

    // copy to device
    cudaMemcpy(device_tmp1, tmp1, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_tmp2, tmp2, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);

    mat_add_kernel<<<blocks, threadsPerBlock>>>(device_tmp1, device_tmp2, r_t, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(r_t, b_r, r_t, hidden_unit, batch_size);
    mat_sigmoid_kernel<<<blocks, threadsPerBlock>>>(r_t, hidden_unit, batch_size);

    // remove after paralleled matrix multiplication
    float* host_r_t = (float*)malloc(batch_size * hidden_unit * sizeof(float));
    cudaMemcpy(host_r_t, r_t, batch_size * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);
    // end of remove
    printf("%.f\n", host_r_t[0]);

    // h_hat = phi(x_t * w_h + (r_t . old_h_t) * u_h + b_h)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_x_t, host_w_h, tmp1, hidden_unit, batch_size, x_width);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(r_t, old_h_t, r_t, hidden_unit, batch_size);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication_kernel(host_r_t, host_u_h, tmp2, hidden_unit, batch_size, hidden_unit);
    float* h_hat;
    cudaMalloc((void **)&h_hat, hidden_unit * batch_size * sizeof(float));

    // copy to device
    cudaMemcpy(device_tmp1, tmp1, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_tmp2, tmp2, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);

    mat_add_kernel<<<blocks, threadsPerBlock>>>(device_tmp1, device_tmp2, h_hat, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(h_hat, b_h, h_hat, hidden_unit, batch_size);
    mat_tanh_kernel<<<blocks, threadsPerBlock>>>(h_hat, hidden_unit, batch_size);

    // new_h_t = (1-z_t).old_h_t + z_t.h_hat
    float* tmp3;
    cudaMalloc((void **)&tmp3, hidden_unit * batch_size * sizeof(float));
    mat_one_sub_kernel<<<blocks, threadsPerBlock>>>(z_t, tmp3, hidden_unit, batch_size);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(tmp3, old_h_t, h_hat, hidden_unit, batch_size);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(z_t, h_hat, h_hat, hidden_unit, batch_size);
    mat_add_kernel<<<blocks, threadsPerBlock>>>(tmp3, h_hat, new_h_t, hidden_unit, batch_size);

}


void
gru_forward_cuda(int batch_size, int x_width, int hidden_unit,
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h) {

    float *device_x_t;
    float *device_old_h_t;
    float *device_new_h_t;
    float *device_w_z;
    float *device_w_r;
    float *device_w_h;
    float *device_u_z;
    float *device_u_r;
    float *device_u_h;
    float *device_b_z;
    float *device_b_r;
    float *device_b_h;

    cudaMalloc((void **)&device_x_t, batch_size * x_width * sizeof(float));
    cudaMalloc((void **)&device_old_h_t, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_new_h_t, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_w_z, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_w_h, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_w_r, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_u_z, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_u_h, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_u_r, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_b_z, 1 * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_b_h, 1 * hidden_unit * sizeof(float));
    cudaMalloc((void **)&device_b_r, 1 * hidden_unit * sizeof(float));

    cudaMemcpy(device_x_t, x_t, batch_size * x_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_old_h_t, old_h_t, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_new_h_t, new_h_t, batch_size * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_w_z, w_z, x_width * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_w_h, w_h, x_width * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_w_r, w_r, x_width * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_u_z, u_z, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_u_h, u_h, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_u_r, u_r, hidden_unit * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b_z, b_z, 1 * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b_h, b_h, 1 * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b_r, b_r, 1 * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);

    gru_forward_kernel(batch_size, x_width, hidden_unit,
                device_x_t, device_old_h_t, device_new_h_t,
                device_w_z, device_w_r, device_w_h,
                device_u_z, device_u_r, device_u_h,
                device_b_z, device_b_r, device_b_h);

    cudaThreadSynchronize();

    cudaMemcpy(new_h_t, device_new_h_t, batch_size * hidden_unit * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(device_x_t);
    cudaFree(device_old_h_t);
    cudaFree(device_new_h_t);
    cudaFree(device_w_z);
    cudaFree(device_w_h);
    cudaFree(device_w_r);
    cudaFree(device_u_z);
    cudaFree(device_u_h);
    cudaFree(device_u_r);
    cudaFree(device_b_z);
    cudaFree(device_b_h);
    cudaFree(device_b_r);
}


void
print_cuda_info() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
