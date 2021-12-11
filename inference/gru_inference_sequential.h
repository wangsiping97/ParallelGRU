#ifndef _GRU_INFERENCE_SEQUANTIAL_H_
#define _GRU_INFERENCE_SEQUANTIAL_H_

#include <cmath>
#include "CycleTimer.h"

void mat_multiplication(float* a, float* b, float* c, int c_width, int c_height, int a_width) {
    for (int i = 0; i < c_height; ++i) {
        for (int j = 0; j < c_width; ++j) {
            for (int k = 0; k < a_width; ++k) {
                c[i * c_width + j] += a[i * a_width + k] * b[k * c_width + j];
            }
        }
    }
}

void mat_add(float* a, float* b, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] = a[i] + b[i];
    }
}

void mat_add_b(float* a, float* b, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] = a[i] + b[i % width];
    }
}

void mat_one_sub(float* a, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] = 1 - a[i];
    }
}

void mat_hadamard(float*a, float* b, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] = a[i] * b[i];
    }
}

void mat_sigmoid(float* a, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        a[i] = 1 / (1 + exp(-1 * a[i]));
    }
}

void mat_tanh(float* a, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        a[i] = tanh(a[i]);
    }
}

// x_t: width: 28, height: batch_size
// old_h_t: width: hidden_unit, height: batch_size
// new_h_t: width: hidden_unit, height: batch_size
// w_z, w_r, w_h: width: hidden_unit, height: 28
// u_z, u_r, u_h: width: hidden_unit, height: hidden_unit
// b_z, b_r, b_h: width: hidden_unit, height: 1
void gru_forward(int batch_size, int x_width, int hidden_unit,
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h) {

    float* tmp1 = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    float* tmp2 = (float*)malloc(hidden_unit * batch_size * sizeof(float));

    // z_t = sigmoid(x_t * w_z + old_h_t * u_z + b_z)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(x_t, w_z, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(old_h_t, u_z, tmp2, hidden_unit, batch_size, hidden_unit);
    float* z_t = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, z_t, hidden_unit, batch_size);
    mat_add_b(z_t, b_z, z_t, hidden_unit, batch_size);
    mat_sigmoid(z_t, hidden_unit, batch_size);

    // r_t = sigmoid(x_t * w_r + old_h_t * u_r + b_r)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(x_t, w_r, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(old_h_t, u_r, tmp2, hidden_unit, batch_size, hidden_unit);
    float* r_t = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, r_t, hidden_unit, batch_size);
    mat_add_b(r_t, b_r, r_t, hidden_unit, batch_size);
    mat_sigmoid(r_t, hidden_unit, batch_size);

    // h_hat = phi(x_t * w_h + (r_t . old_h_t) * u_h + b_h)
    memset(tmp1, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(x_t, w_h, tmp1, hidden_unit, batch_size, x_width);
    mat_hadamard(r_t, old_h_t, r_t, hidden_unit, batch_size);
    memset(tmp2, 0, hidden_unit * batch_size * sizeof(float));
    mat_multiplication(r_t, u_h, tmp2, hidden_unit, batch_size, hidden_unit);
    float* h_hat = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, h_hat, hidden_unit, batch_size);
    mat_add_b(h_hat, b_h, h_hat, hidden_unit, batch_size);
    mat_tanh(h_hat, hidden_unit, batch_size);

    // new_h_t = (1-z_t).old_h_t + z_t.h_hat
    float* tmp3 = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_one_sub(z_t, tmp3, hidden_unit, batch_size);
    mat_hadamard(tmp3, old_h_t, tmp3, hidden_unit, batch_size);
    mat_hadamard(z_t, h_hat, h_hat, hidden_unit, batch_size);
    mat_add(tmp3, h_hat, new_h_t, hidden_unit, batch_size);

}

void inference(int num_data, int batch_size, int window_size, int vec_len, int hidden_unit, float step_size,
                float* h_t, float* h_t_new,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h,
                float* dense, float* predict, float* arr_data, int m, int n) {
    
    double startTime = CycleTimer::currentSeconds();

    // One iteration, loop through all data point
    for (int i = 0; i < num_data; i += batch_size) {

        // batch_size * (num_data * vec_len)
        int start_i = i;
        int end_i = std::min(num_data, i + batch_size);
        int batch = end_i - start_i;
        
        float* x_t = (float*)calloc(batch_size * vec_len, sizeof(float));

        // for each time step
        for (int j = 0; j < window_size; j++) {

            // Construct x_t
            for (int _m = 0; _m < batch; _m++) {
                for (int _n = 0; _n < vec_len; _n++) {
                    x_t[_m * vec_len + _n] = arr_data[(start_i + _m) * n + j + _n];
                }
            }

            // one forward iteration: 
            gru_forward(batch_size, vec_len, hidden_unit, x_t, h_t, h_t_new, 
                        w_z, w_r, w_h, u_z, u_r, u_h, b_z, b_r, b_h); 

            // update h_t for next round
            memcpy(h_t, h_t_new, batch_size * hidden_unit * sizeof(float));
            memset(h_t_new, 0.f, batch_size * hidden_unit * sizeof(float));
        }

        // inference
        mat_multiplication(h_t, dense, predict, 1, batch_size, hidden_unit);

    }
    
    double endTime = CycleTimer::currentSeconds();
    printf("CPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));
}

#endif // _GRU_INFERENCE_SEQUANTIAL_H_