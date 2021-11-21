#include <iostream>
#include <cmath>

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

void gru_forward(int batch_size, int x_width, 
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h) {

    // initialize new_h_t
    memset(new_h_t, 0, sizeof(new_h_t));

    float* tmp1 = (float*)malloc(x_width * batch_size * sizeof(float));
    float* tmp2 = (float*)malloc(x_width * batch_size * sizeof(float));

    // z_t = sigmoid(w_z * x_t + u_z * old_h_t + b_z)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(w_z, x_t, tmp1, x_width, batch_size, batch_size);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_z, old_h_t, tmp2, x_width, batch_size, batch_size);
    float* z_t = (float*)malloc(x_width * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, z_t, x_width, batch_size);
    mat_add(z_t, b_z, z_t, x_width, batch_size);
    mat_sigmoid(z_t, x_width, batch_size);

    // r_t = sigmoid(w_r * x_t + u_r * old_h_t + b_r)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(w_r, x_t, tmp1, x_width, batch_size, batch_size);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_r, old_h_t, tmp2, x_width, batch_size, batch_size);
    float* r_t = (float*)malloc(x_width * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, r_t, x_width, batch_size);
    mat_add(r_t, b_r, r_t, x_width, batch_size);
    mat_sigmoid(r_t, x_width, batch_size);

    // h_hat = phi(w_h*x_t + u_h(r_t . old_h_t) + b_h)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(w_h, x_t, tmp1, x_width, batch_size, batch_size);
    mat_hadamard(r_t, old_h_t, r_t, x_width, batch_size);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_h, r_t, tmp2, x_width, batch_size, batch_size);
    float* h_hat = (float*)malloc(x_width * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, h_hat, x_width, batch_size);
    mat_add(h_hat, b_h, h_hat, x_width, batch_size);
    mat_tanh(h_hat, x_width, batch_size);

    // new_h_t = (1-z_t).old_h_t + z_t.h_hat
    float* tmp3 = (float*)malloc(x_width * batch_size * sizeof(float));
    mat_one_sub(z_t, tmp3, x_width, batch_size);
    mat_hadamard(tmp3, old_h_t, tmp3, x_width, batch_size);
    mat_hadamard(z_t, h_hat, h_hat, x_width, batch_size);
    mat_add(tmp3, h_hat, new_h_t, x_width, batch_size);

    // free temp arrays
    free(tmp1);
    free(tmp2);
    free(tmp3);

}