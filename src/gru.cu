#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define threadsPerBlock 512

extern float toBW(int bytes, float sec);
extern float calculate_loss(int batch, float *y, float *predict);

__global__ void
copy_data_kernel(float* x_t, int x_height, int x_width, float* data, int m, int n, int start_i, int j) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = index / x_width;
    int index_x = index % x_width;
    if (index < x_height * x_width) {
        x_t[index] = data[(start_i + index_y) * n + j + index_x];
    }
}

__global__ void
mat_init_zeros_kernel(float* a, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        a[index] = 0.0;
    }
}

__global__ void
mat_copy_kernel(float* dest, float* src, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        dest[index] = src[index];
    }
}

__global__ void
mat_multiplication_kernel(float* a, float* b, float* c, int c_width, int c_height, int a_width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = index / c_width;
    int index_x = index % c_width;
    if (index < c_width * c_height) {
        float tmp = 0;
        for (int i = 0; i < a_width; ++i) {
            tmp += a[index_y * a_width + i] * b[i * c_width + index_x];
        }
        c[index_y * c_width + index_x] = tmp;
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

__global__ void
mat_sub_kernel(float* a, float* b, float* res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = a[index] - b[index];
}

__global__ void 
mat_div_kernel(float *a, float *b, float *res, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        res[index] = a[index] / b[index];
}

__global__ void
mat_transpose_kernel(float *a, float* res, int a_width, int a_height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // res's index
    int index_y = index / a_height; 
    int index_x = index % a_height;

    // res[index_y][index_x] = a[index_x][index_y]
    if (index < a_width * a_height)
        res[index] = a[index_x * a_width + index_y];
}

__global__ void
update_variable_kernel(float* a, float* grad, int width, int height, float step_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < width * height) 
        a[index] -= step_size * grad[index];
}

__global__ void
sum_over_rows_kernel(float* a, float* b, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // b's index
    if (index < width) {
        b[index] = 0;
        for (int i = 0; i < height; ++i) {
            b[index] += a[i * width + index];
        }
    }
}

__global__ void
update_grad_predict_kernel(float* predict, float* device_y, float* grad_predict, float loss, int batch) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch) {
        grad_predict[index] = predict[index] - device_y[index];
        grad_predict[index] *= 2 * loss / batch;
    }
}

__global__ void
update_old_h_t_kernel(float* h_t, float* H_1, int j, int batch_size, int hidden_unit) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < batch_size * hidden_unit) 
        h_t[index] = H_1[(j + 1) * batch_size * hidden_unit];
}

int computeBlocks(int length) {
    return (length + threadsPerBlock - 1) / threadsPerBlock;
}

void Print_Device(float* device_data, int length) {
    printf("Print...\n");
    float* test = (float*)calloc(length, sizeof(float));
    cudaMemcpy(test, device_data, length * sizeof(float), cudaMemcpyDeviceToHost);
    for (int k = 0; k < length; ++k) {
        printf("%.12f ", test[k]);
    }
    printf("\n");
}

void update_dense_and_grad_h_t_kernel(int start_i, int batch, int hidden_unit, int batch_size, int step_size, float loss,
                                    float* dense, float* grad_h_t, float* predict, float* h_t, float* device_y) {
    
    float* grad_dense;
    float* grad_predict;

    cudaMalloc((void **)&grad_dense, hidden_unit * 1 * sizeof(float));
    cudaMalloc((void **)&grad_predict, batch_size * 1 * sizeof(float));

    // d loss / d predict
    update_grad_predict_kernel<<<computeBlocks(batch), threadsPerBlock>>>(predict, device_y, grad_predict, loss, batch);

    // d loss / d final_h_t
    mat_multiplication_kernel<<<computeBlocks(hidden_unit * batch_size), threadsPerBlock>>>(grad_predict, dense, grad_h_t, hidden_unit, batch_size, 1);

    // d loss / d dense
    float* h_t_T;
    cudaMalloc((void **)&h_t_T, batch_size * hidden_unit * sizeof(float));
    mat_transpose_kernel<<<computeBlocks(hidden_unit * batch_size), threadsPerBlock>>>(h_t, h_t_T, batch_size, hidden_unit);
    mat_multiplication_kernel<<<computeBlocks(1 * hidden_unit), threadsPerBlock>>>(h_t_T, grad_predict, grad_dense, 1, hidden_unit, batch_size);

    update_variable_kernel<<<computeBlocks(hidden_unit * 1), threadsPerBlock>>>(dense, grad_dense, hidden_unit, 1, step_size);

    cudaFree(h_t_T);
    cudaFree(grad_dense);
    cudaFree(grad_predict);

}

// x_t: width: 28, height: batch_size
// old_h_t: width: hidden_unit, height: batch_size
// new_h_t: width: hidden_unit, height: batch_size
// w_z, w_r, w_h: width: hidden_unit, height: 28
// u_z, u_r, u_h: width: hidden_unit, height: hidden_unit
// b_z, b_r, b_h: width: hidden_unit, height: 1
void gru_forward_kernel(int timestep, float* Z, float* H_hat, float* H_1, float* R, 
                int batch_size, int x_width, int hidden_unit,
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h) {

    const int blocks = (hidden_unit * batch_size + threadsPerBlock - 1) / threadsPerBlock;

    float* tmp1;
    float* tmp2;
    cudaMalloc((void **)&tmp1, hidden_unit * batch_size * sizeof(float));
    cudaMalloc((void **)&tmp2, hidden_unit * batch_size * sizeof(float));

    // z_t = sigmoid(x_t * w_z + old_h_t * u_z + b_z)
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(x_t, w_z, tmp1, hidden_unit, batch_size, x_width);
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(old_h_t, u_z, tmp2, hidden_unit, batch_size, hidden_unit);

    float* z_t;
    cudaMalloc((void **)&z_t, hidden_unit * batch_size * sizeof(float));
    mat_add_kernel<<<blocks, threadsPerBlock>>>(tmp1, tmp2, z_t, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(z_t, b_z, z_t, hidden_unit, batch_size);
    mat_sigmoid_kernel<<<blocks, threadsPerBlock>>>(z_t, hidden_unit, batch_size);

    // r_t = sigmoid(x_t * w_r + old_h_t * u_r + b_r)
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(x_t, w_r, tmp1, hidden_unit, batch_size, x_width);
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(old_h_t, u_r, tmp2, hidden_unit, batch_size, hidden_unit);
    
    float* r_t;
    cudaMalloc((void **)&r_t, hidden_unit * batch_size * sizeof(float));

    mat_add_kernel<<<blocks, threadsPerBlock>>>(tmp1, tmp2, r_t, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(r_t, b_r, r_t, hidden_unit, batch_size);
    mat_sigmoid_kernel<<<blocks, threadsPerBlock>>>(r_t, hidden_unit, batch_size);

    // h_hat = phi(x_t * w_h + (r_t . old_h_t) * u_h + b_h)
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(x_t, w_h, tmp1, hidden_unit, batch_size, x_width);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(r_t, old_h_t, r_t, hidden_unit, batch_size);
    mat_multiplication_kernel<<<blocks, threadsPerBlock>>>(r_t, u_h, tmp2, hidden_unit, batch_size, hidden_unit);

    float* h_hat;
    cudaMalloc((void **)&h_hat, hidden_unit * batch_size * sizeof(float));

    mat_add_kernel<<<blocks, threadsPerBlock>>>(tmp1, tmp2, h_hat, hidden_unit, batch_size); 
    mat_add_b_kernel<<<blocks, threadsPerBlock>>>(h_hat, b_h, h_hat, hidden_unit, batch_size);
    mat_tanh_kernel<<<blocks, threadsPerBlock>>>(h_hat, hidden_unit, batch_size);

    // new_h_t = (1-z_t).old_h_t + z_t.h_hat
    float* tmp3;
    cudaMalloc((void **)&tmp3, hidden_unit * batch_size * sizeof(float));
    mat_one_sub_kernel<<<blocks, threadsPerBlock>>>(z_t, tmp3, hidden_unit, batch_size);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(tmp3, old_h_t, tmp3, hidden_unit, batch_size);
    mat_hadamard_kernel<<<blocks, threadsPerBlock>>>(z_t, h_hat, h_hat, hidden_unit, batch_size);
    mat_add_kernel<<<blocks, threadsPerBlock>>>(tmp3, h_hat, new_h_t, hidden_unit, batch_size);

    // update Z, R, H_hat, H_1
    mat_copy_kernel<<<blocks, threadsPerBlock>>>(Z, z_t, hidden_unit * batch_size);
    mat_copy_kernel<<<blocks, threadsPerBlock>>>(R, r_t, hidden_unit * batch_size);
    mat_copy_kernel<<<blocks, threadsPerBlock>>>(H_hat, h_hat, hidden_unit * batch_size);
    mat_copy_kernel<<<blocks, threadsPerBlock>>>(H_1, old_h_t, hidden_unit * batch_size);

}

void gru_backward_kernel(int i, int vec_len, int hidden_unit, int batch_size, float step_size,
                        float* grad_h_t, float* h_t, float* x_t, 
                        float* u_z, float* u_r, float* u_h,
                        float* w_z, float* w_r, float* w_h,
                        float* b_z, float* b_r, float* b_h,
                        float* Grad_u_z, float* Grad_u_r, float* Grad_u_h,
                        float* grad_u_z, float* grad_u_r, float* grad_u_h, 
                        float* grad_w_z, float* grad_w_r, float* grad_w_h,
                        float* grad_b_z, float* grad_b_r, float* grad_b_h,
                        float* grad_r_t, float* grad_r_t_before_sigmoid,
                        float* grad_z_t, float* grad_z_t_before_sigmoid,
                        float* grad_h_hat, float* grad_h_hat_before_sigmoid,
                        float* grad_h_t_1,
                        float* Z, float* R, float* H_hat, float* H_1) {
    
    // for current timestep:
    // d loss / d z_t
    int block_h_b = computeBlocks(hidden_unit * batch_size);
    float *tmp1;
    cudaMalloc((void **)&tmp1, batch_size * hidden_unit * sizeof(float));
    mat_sub_kernel<<<block_h_b,threadsPerBlock>>>(H_hat, h_t, tmp1, hidden_unit, batch_size);
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t, tmp1, grad_z_t, hidden_unit, batch_size);

    // d loss / d h_t_1
    mat_one_sub_kernel<<<block_h_b, threadsPerBlock>>>(Z, tmp1, hidden_unit, batch_size);
    mat_add_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t, tmp1, grad_h_t_1, hidden_unit, batch_size);

    // d loss / d h_hat
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t, Z, grad_h_hat, hidden_unit, batch_size);

    // d loss / d h_hat_before_sigmoid
    mat_one_sub_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_hat, tmp1, hidden_unit, batch_size);
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_hat, tmp1, grad_h_hat_before_sigmoid, hidden_unit, batch_size);

    // d loss / Wh
    int block_w = computeBlocks(vec_len * hidden_unit);
    float* tmp2;
    cudaMalloc((void **)&tmp2, vec_len * batch_size * sizeof(float));
    mat_transpose_kernel<<<computeBlocks(vec_len * batch_size), threadsPerBlock>>>(x_t, tmp2, vec_len, batch_size);
    mat_multiplication_kernel<<<block_w, threadsPerBlock>>>(tmp2, grad_h_hat_before_sigmoid, grad_w_h, hidden_unit, vec_len, batch_size);

    // d loss / u_h
    int block_u = computeBlocks(hidden_unit * hidden_unit);
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(R, H_1, tmp1, hidden_unit, batch_size);
    float* tmp3; 
    cudaMalloc((void **)&tmp3, hidden_unit * batch_size * sizeof(float));
    mat_transpose_kernel<<<block_h_b, threadsPerBlock>>>(tmp1, tmp3, hidden_unit, batch_size);
    mat_multiplication_kernel<<<block_u, threadsPerBlock>>>(tmp3, grad_h_hat_before_sigmoid, grad_u_h, hidden_unit, hidden_unit, batch_size);

    // d loss / b_h
    int block_b = computeBlocks(hidden_unit);
    sum_over_rows_kernel<<<block_b, threadsPerBlock>>>(grad_h_hat_before_sigmoid, grad_b_h, hidden_unit, batch_size);

    // d loss / r_t
    float* tmp4;
    cudaMalloc((void **)&tmp4, hidden_unit * hidden_unit * sizeof(float));
    mat_transpose_kernel<<<block_u, threadsPerBlock>>>(u_h, tmp4, hidden_unit, hidden_unit);
    mat_multiplication_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_hat_before_sigmoid, tmp4, tmp1, hidden_unit, batch_size, hidden_unit);
    mat_div_kernel<<<block_h_b, threadsPerBlock>>>(tmp1, H_1, grad_r_t, hidden_unit, batch_size);

    // d loss / h_t
    mat_div_kernel<<<block_h_b, threadsPerBlock>>>(tmp1, R, tmp1, hidden_unit, batch_size);
    mat_add_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t_1, tmp1, grad_h_t_1, hidden_unit, batch_size);

    // d loss / d r_t_before_sigmoid
    mat_one_sub_kernel<<<block_h_b, threadsPerBlock>>>(grad_r_t, tmp1, hidden_unit, batch_size);
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(grad_r_t, tmp1, grad_r_t_before_sigmoid, hidden_unit, batch_size);

    // d loss / d w_r
    mat_multiplication_kernel<<<block_w, threadsPerBlock>>>(tmp2, grad_r_t_before_sigmoid, grad_w_r, hidden_unit, vec_len, batch_size);

    // d loss / d u_r
    mat_transpose_kernel<<<block_h_b, threadsPerBlock>>>(H_1, tmp3, hidden_unit, batch_size);
    mat_multiplication_kernel<<<block_u, threadsPerBlock>>>(tmp3, grad_r_t_before_sigmoid, grad_u_r, hidden_unit, hidden_unit, batch_size);

    // d loss / d b_r
    sum_over_rows_kernel<<<block_b, threadsPerBlock>>>(grad_r_t_before_sigmoid, grad_b_r, hidden_unit, batch_size);

    // d loss / d h_t
    mat_transpose_kernel<<<block_u, threadsPerBlock>>>(u_r, tmp4, hidden_unit, hidden_unit);
    mat_multiplication_kernel<<<block_h_b, threadsPerBlock>>>(grad_r_t_before_sigmoid, tmp4, tmp1, hidden_unit, batch_size, hidden_unit);
    // if (i == 0) Print_Device(grad_r_t_before_sigmoid, hidden_unit * batch_size);

    mat_add_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t_1, tmp1, grad_h_t_1, hidden_unit, batch_size);
    // if(i == 0) Print_Device(grad_h_t_1, hidden_unit * batch_size);

    // d loss / d z_t_before_sigmoid
    mat_one_sub_kernel<<<block_h_b, threadsPerBlock>>>(grad_z_t, tmp1, hidden_unit, batch_size);
    mat_hadamard_kernel<<<block_h_b, threadsPerBlock>>>(grad_z_t, tmp1, grad_z_t_before_sigmoid, hidden_unit, batch_size);

    // d loss / d w_z
    mat_multiplication_kernel<<<block_w, threadsPerBlock>>>(tmp2, grad_z_t_before_sigmoid, grad_w_z, hidden_unit, vec_len, batch_size);

    // d loss / d u_z
    mat_multiplication_kernel<<<block_u, threadsPerBlock>>>(tmp3, grad_z_t_before_sigmoid, grad_u_z, hidden_unit, hidden_unit, batch_size);

    // d loss / d b_z
    sum_over_rows_kernel<<<block_b, threadsPerBlock>>>(grad_z_t_before_sigmoid, grad_b_z, hidden_unit, batch_size);

    // d loss / d h_t
    mat_transpose_kernel<<<block_u, threadsPerBlock>>>(u_z, tmp4, hidden_unit, hidden_unit);
    mat_multiplication_kernel<<<block_h_b, threadsPerBlock>>>(grad_z_t_before_sigmoid, tmp4, tmp1, hidden_unit, batch_size, hidden_unit);
    mat_add_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t_1, tmp1, grad_h_t_1, hidden_unit, batch_size);

    // loss for next timestep
    mat_copy_kernel<<<block_h_b, threadsPerBlock>>>(grad_h_t_1, grad_h_t, batch_size * hidden_unit);

    // cumulate gradient for all timesteps
    update_variable_kernel<<<block_w, threadsPerBlock>>>(w_z, grad_w_z, hidden_unit, vec_len, step_size);
    update_variable_kernel<<<block_w, threadsPerBlock>>>(w_r, grad_w_r, hidden_unit, vec_len, step_size);
    update_variable_kernel<<<block_w, threadsPerBlock>>>(w_h, grad_w_h, hidden_unit, vec_len, step_size);

    mat_add_kernel<<<block_u, threadsPerBlock>>>(Grad_u_z, grad_u_z, Grad_u_z, hidden_unit, hidden_unit);
    mat_add_kernel<<<block_u, threadsPerBlock>>>(Grad_u_r, grad_u_r, Grad_u_r, hidden_unit, hidden_unit);
    mat_add_kernel<<<block_u, threadsPerBlock>>>(Grad_u_h, grad_u_h, Grad_u_h, hidden_unit, hidden_unit);

    update_variable_kernel<<<block_b, threadsPerBlock>>>(b_z, grad_b_z, hidden_unit, 1, step_size);
    update_variable_kernel<<<block_b, threadsPerBlock>>>(b_r, grad_b_r, hidden_unit, 1, step_size);
    update_variable_kernel<<<block_b, threadsPerBlock>>>(b_h, grad_b_h, hidden_unit, 1, step_size);

    cudaFree(tmp1);
    cudaFree(tmp2);
    cudaFree(tmp3);
    cudaFree(tmp4);
}

void run_model_cuda(int num_data, int batch_size, int window_size, int x_width, int hidden_unit, float step_size,
                        float* old_h_t, float* new_h_t,
                        float* w_z, float* w_r, float* w_h,
                        float* u_z, float* u_r, float* u_h,
                        float* b_z, float* b_r, float* b_h,
                        float* dense, float* predict, float* arr_data, int m, int n, float* y, int iter) {

    double startTime = CycleTimer::currentSeconds();

    // allocate variables
    float *device_data;
    float *device_y;

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

    float *device_dense;
    float *device_predict;

    cudaMalloc((void **)&device_data, m * n * sizeof(float));
    cudaMalloc((void **)&device_y, num_data * sizeof(float));

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

    cudaMalloc((void **)&device_dense, hidden_unit * 1 * sizeof(float));
    cudaMalloc((void **)&device_predict, batch_size * 1 * sizeof(float));

    cudaMemcpy(device_data, arr_data, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, y, num_data * sizeof(float), cudaMemcpyHostToDevice);

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
    cudaMemcpy(device_dense, dense, 1 * hidden_unit * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_predict, predict, 1 * batch_size * sizeof(float), cudaMemcpyHostToDevice);

    // allocate gradient variables
    float* Z;
    float* R;
    float* H_hat;
    float* H_1;
    
    float* grad_h_t;
    float* Grad_u_z;
    float* Grad_u_r;
    float* Grad_u_h;
    
    float* grad_w_z;
    float* grad_w_r;
    float* grad_w_h;
    float* grad_u_z;
    float* grad_u_r;
    float* grad_u_h;
    float* grad_b_z;
    float* grad_b_r;
    float* grad_b_h;
    float* grad_r_t;
    float* grad_r_t_before_sigmoid;
    float* grad_z_t;
    float* grad_z_t_before_sigmoid;
    float* grad_h_hat;
    float* grad_h_hat_before_sigmoid;
    float* grad_h_t_1;

    cudaMalloc((void **)&Z, window_size * batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&R, window_size * batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&H_hat, window_size * batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&H_1, window_size * batch_size * hidden_unit * sizeof(float));
    
    cudaMalloc((void **)&grad_h_t, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&Grad_u_z, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&Grad_u_r, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&Grad_u_h, hidden_unit * hidden_unit * sizeof(float));

    cudaMalloc((void **)&grad_w_z, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_w_r, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_w_h, x_width * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_u_z, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_u_r, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_u_h, hidden_unit * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_b_z, hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_b_r, hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_b_h, hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_r_t, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_r_t_before_sigmoid, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_z_t, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_z_t_before_sigmoid, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_h_hat, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_h_hat_before_sigmoid, batch_size * hidden_unit * sizeof(float));
    cudaMalloc((void **)&grad_h_t_1, batch_size * hidden_unit * sizeof(float));

    const int blocks_h = (hidden_unit * batch_size + threadsPerBlock - 1) / threadsPerBlock;
    const int blocks_predict = (batch_size + threadsPerBlock - 1) / threadsPerBlock;
    const int blocks_x = (x_width * batch_size + threadsPerBlock - 1) / threadsPerBlock;
    int block_w = computeBlocks(x_width * hidden_unit);
    int block_u = computeBlocks(hidden_unit * hidden_unit);
    int block_b = computeBlocks(hidden_unit);

    double forwardTime = 0;
    double inferenceTime = 0;
    double backwardTime = 0;
    double iterStartTime = CycleTimer::currentSeconds();

    for (int num_iter = 0; num_iter < iter; num_iter++) {
        printf("begin iter %d\n", num_iter);
        // One iteration, loop through all data point
        for (int i = 0; i < num_data; i += batch_size) {

            // batch_size * (num_data * x_width)
            int start_i = i;
            int end_i = min(num_data, i + batch_size);
            int batch = end_i - start_i;

            // for each time step
            double forwardStartTime = CycleTimer::currentSeconds();
            for (int j = 0; j < window_size; j++) {

                copy_data_kernel<<<blocks_x, threadsPerBlock>>>(device_x_t, batch, x_width, device_data, m, n, start_i, j);

                // one forward iteration: 
                gru_forward_kernel(j, &Z[j*hidden_unit*batch_size], &H_hat[j*hidden_unit*batch_size], 
                    &H_1[j*hidden_unit*batch_size], &R[j*hidden_unit*batch_size], 
                    batch_size, x_width, hidden_unit, device_x_t, device_old_h_t, device_new_h_t, 
                    device_w_z, device_w_r, device_w_h, device_u_z, device_u_r, device_u_h, device_b_z, device_b_r, device_b_h); 

                // update h_t for the next round
                mat_copy_kernel<<<blocks_h, threadsPerBlock>>>(device_old_h_t, device_new_h_t, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(device_new_h_t, batch_size * hidden_unit);

            }
            double forwardEndTime = CycleTimer::currentSeconds();
            forwardTime += forwardEndTime - forwardStartTime;

            // inference
            double inferenceStartTime = CycleTimer::currentSeconds();
            mat_multiplication_kernel<<<blocks_predict, threadsPerBlock>>>(device_dense, device_old_h_t, device_predict, batch_size, 1, hidden_unit);

            cudaMemcpy(predict, device_predict, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            // calculate loss
            float loss = calculate_loss(batch, &y[start_i], predict);
            // printf("loss is %.6f\n", loss);
            double inferenceEndTime = CycleTimer::currentSeconds();
            inferenceTime += inferenceEndTime - inferenceStartTime;

            double backwardTimeStart = CycleTimer::currentSeconds();
            // reset gradients
            mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_h_t, batch_size * hidden_unit);

            // calculate gradients for predice, dense, and h_t
            update_dense_and_grad_h_t_kernel(start_i, batch, hidden_unit, batch_size, step_size, loss,
                                            device_dense, grad_h_t, device_predict, device_old_h_t, &device_y[start_i]);
            
            // calculate gradient for each time_step
            for (int j = window_size-1; j >= 1; j--) {
                
                // reset gradients
                mat_init_zeros_kernel<<<block_w, threadsPerBlock>>>(grad_w_z, x_width * hidden_unit);
                mat_init_zeros_kernel<<<block_w, threadsPerBlock>>>(grad_w_h, x_width * hidden_unit);
                mat_init_zeros_kernel<<<block_w, threadsPerBlock>>>(grad_w_r, x_width * hidden_unit);
                mat_init_zeros_kernel<<<block_u, threadsPerBlock>>>(grad_u_z, hidden_unit * hidden_unit);
                mat_init_zeros_kernel<<<block_u, threadsPerBlock>>>(grad_u_h, hidden_unit * hidden_unit);
                mat_init_zeros_kernel<<<block_u, threadsPerBlock>>>(grad_u_r, hidden_unit * hidden_unit);
                mat_init_zeros_kernel<<<block_b, threadsPerBlock>>>(grad_b_z, hidden_unit);
                mat_init_zeros_kernel<<<block_b, threadsPerBlock>>>(grad_b_r, hidden_unit);
                mat_init_zeros_kernel<<<block_b, threadsPerBlock>>>(grad_b_h, hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_z_t, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_z_t_before_sigmoid, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_r_t, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_r_t_before_sigmoid, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_h_hat, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_h_hat_before_sigmoid, batch_size * hidden_unit);
                mat_init_zeros_kernel<<<blocks_h, threadsPerBlock>>>(grad_h_t_1, batch_size * hidden_unit);

                // Construct x_t
                copy_data_kernel<<<blocks_x, threadsPerBlock>>>(device_x_t, batch, x_width, device_data, m, n, start_i, j);

                // Update h_t
                if (j != window_size - 1) {
                    update_old_h_t_kernel<<<blocks_h, threadsPerBlock>>>(device_old_h_t, H_1, j, batch_size, hidden_unit);
                }

                // call gru_backward
                gru_backward_kernel(i, x_width, hidden_unit, batch_size, step_size,
                                    grad_h_t, device_old_h_t, device_x_t, 
                                    device_u_z, device_u_r, device_u_h,
                                    device_w_z, device_w_r, device_w_h,
                                    device_b_z, device_b_r, device_b_h, 
                                    Grad_u_z, Grad_u_r, Grad_u_h, 
                                    grad_u_z, grad_u_r, grad_u_h,
                                    grad_w_z, grad_w_r, grad_w_h,
                                    grad_b_z, grad_b_r, grad_b_h,
                                    grad_r_t, grad_r_t_before_sigmoid,
                                    grad_z_t, grad_z_t_before_sigmoid,
                                    grad_h_hat, grad_h_hat_before_sigmoid,
                                    grad_h_t_1,
                                    &Z[j*hidden_unit*batch_size],
                                    &R[j*hidden_unit*batch_size],
                                    &H_hat[j*hidden_unit*batch_size],
                                    &H_1[j*hidden_unit*batch_size]);
            }

            // update variables
            int update_block = computeBlocks(hidden_unit * hidden_unit);
            update_variable_kernel<<<update_block, threadsPerBlock>>>(device_u_z, Grad_u_z, hidden_unit, hidden_unit, step_size);
            update_variable_kernel<<<update_block, threadsPerBlock>>>(device_u_r, Grad_u_r, hidden_unit, hidden_unit, step_size);
            update_variable_kernel<<<update_block, threadsPerBlock>>>(device_u_h, Grad_u_h, hidden_unit, hidden_unit, step_size);

            double backwardTimeEnd = CycleTimer::currentSeconds();
            backwardTime += backwardTimeEnd - backwardTimeStart;

        }
    }

    double iterEndTime = CycleTimer::currentSeconds();

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
    cudaFree(device_dense);
    cudaFree(device_predict);

    cudaFree(Z);
    cudaFree(R);
    cudaFree(H_hat);
    cudaFree(H_1);
    cudaFree(grad_h_t);
    cudaFree(Grad_u_z);
    cudaFree(Grad_u_r);
    cudaFree(Grad_u_h);

    cudaFree(grad_w_z);
    cudaFree(grad_w_r);
    cudaFree(grad_w_h);
    cudaFree(grad_u_z);
    cudaFree(grad_u_r);
    cudaFree(grad_u_h);
    cudaFree(grad_b_z);
    cudaFree(grad_b_r);
    cudaFree(grad_b_h);
    cudaFree(grad_r_t);
    cudaFree(grad_r_t_before_sigmoid);
    cudaFree(grad_z_t);
    cudaFree(grad_z_t_before_sigmoid);
    cudaFree(grad_h_hat);
    cudaFree(grad_h_hat_before_sigmoid);
    cudaFree(grad_h_t_1);

    double endTime = CycleTimer::currentSeconds();
    printf("GPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));
    printf("GPU Compute: %.3f ms\n", 1000.f * (iterEndTime - iterStartTime));
    printf("GPU Forward: %.3f ms\n", 1000.f * (forwardTime));
    printf("GPU Inference: %.3f ms\n", 1000.f * (inferenceTime));
    printf("GPU Backward: %.3f ms\n", 1000.f * (backwardTime));
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
