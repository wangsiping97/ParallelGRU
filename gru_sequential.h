#ifndef _GRU_SEQUANTIAL_H_
#define _GRU_SEQUANTIAL_H_

#include <cmath>
#include "CycleTimer.h"

using namespace std;

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

void mat_sub(float* a, float* b, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] = a[i] - b[i];
    }
}

void mat_div(float *a, float *b, float *res, int width, int height) {
    for (int i = 0; i < width*height; ++i) {
        res[i] = a[i] / b[i];
    }
}

float* mat_transpose(float *a, int width, int height) {
    float *tmp = (float*)malloc(width * height * sizeof(float));

    // transpose a into tmp
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            tmp[j * height + i] = a[i * width + j];
        }
    }

    return tmp;
}

void update_variable(float *a, float *grad, int width, int height, float step_size) {
    for (int i = 0; i < width * height; i++) {
        a[i] -= step_size * grad[i];
    }
}

// sum rows of a into b
void sum_over_rows(float *a, float *b, int width, int height) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            b[j] += a[i * width + j];
        }
    }
}

void Print(float* data, int length) {
    for (int i = 0; i < length; ++i) 
        printf("%.12f ", data[i]);
    printf("\n");
}

// x_t: width: 28, height: batch_size
// old_h_t: width: hidden_unit, height: batch_size
// new_h_t: width: hidden_unit, height: batch_size
// w_z, w_r, w_h: width: hidden_unit, height: 28
// u_z, u_r, u_h: width: hidden_unit, height: hidden_unit
// b_z, b_r, b_h: width: hidden_unit, height: 1
void gru_forward(int timestep, float* Z, float* H_hat, float* H_1, 
                float *R, 
                int batch_size, int x_width, int hidden_unit,
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

    for (int i = 0; i < batch_size * hidden_unit; i++) {
        Z[timestep*hidden_unit*batch_size + i] = z_t[i];
        R[timestep*hidden_unit*batch_size + i] = r_t[i];
        H_hat[timestep*hidden_unit*batch_size + i] = h_hat[i];
        H_1[timestep*hidden_unit*batch_size + i] = old_h_t[i];
    }

    free(tmp1);
    free(tmp2);
    free(tmp3);
    free(z_t);
    free(r_t);
    free(h_hat);
}

float calculate_loss(int batch, float *y, float *predict) {
    float loss = 0.f;
    float diff;

    // calculate mean squared difference between y and predict for this batch
    for (int _m = 0; _m < batch; _m++) {
        diff = y[_m] - predict[_m];
        loss += pow(diff, 2);
    }

    loss = loss / batch;
    return loss;
}

void update_dense_and_grad_h_t(int batch, int hidden_unit, int batch_size, int step_size, float loss,
                                float *dense, float *grad_h_t, float *predict, float *h_t, float *y) {

    float* grad_dense = (float*)calloc(hidden_unit * 1, sizeof(float));
    float* grad_predict = (float*)calloc(batch_size * 1, sizeof(float));

    // d loss / d predict
    for (int _m = 0; _m < batch; _m++) {
        grad_predict[_m] = predict[_m] - y[_m];
        grad_predict[_m] *= 2 * loss / batch;
    }

    // d loss / d final_h_t
    mat_multiplication(grad_predict, dense, grad_h_t, hidden_unit, batch_size, 1);

    // d loss / d dense
    float *h_t_T = mat_transpose(h_t, batch_size, hidden_unit);
    mat_multiplication(h_t_T, grad_predict, grad_dense, 1, hidden_unit, batch_size);

    update_variable(dense, grad_dense, hidden_unit, 1, step_size);

    free(h_t_T);
    free(grad_dense);
    free(grad_predict);
}

void gru_backward(int i_start, int vec_len, int hidden_unit, int batch_size, float step_size,
                  float *grad_h_t, float *h_t,
                  float *x_t, 
                  float *u_z, float *u_r, float *u_h, 
                  float *w_z, float *w_r, float *w_h, 
                  float *b_z, float *b_r, float *b_h,
                  float *Grad_u_z, float *Grad_u_r, float *Grad_u_h,
                  float *Z, float* R, float* H_hat, float* H_1) {

    // if(i == 0) Print(Z, hidden_unit*batch_size);
    
    // reset gradients for current timestep
    float *z_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float *r_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float *h_hat = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float *h_t_1 = (float*)calloc(batch_size * hidden_unit, sizeof(float));

    for (int i = 0; i < batch_size * hidden_unit; i++) {
        z_t[i] = Z[i];
        r_t[i] = R[i];
        h_hat[i] = H_hat[i];
        h_t_1[i] = H_1[i];
    }

    float* grad_w_z = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* grad_w_r = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* grad_w_h = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* grad_u_z = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* grad_u_r = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* grad_u_h = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* grad_b_z = (float*)calloc(hidden_unit, sizeof(float));
    float* grad_b_r = (float*)calloc(hidden_unit, sizeof(float));
    float* grad_b_h = (float*)calloc(hidden_unit, sizeof(float));   
    float* grad_r_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* grad_r_t_before_sigmoid = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* grad_z_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* grad_z_t_before_sigmoid = (float*)calloc(batch_size * hidden_unit, sizeof(float)); 
    float* grad_h_hat = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* grad_h_hat_before_sigmoid = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* grad_h_t_1 = (float*)calloc(batch_size * hidden_unit, sizeof(float));

    // for current timestep:
    // d loss / d z_t
    float *tmp = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    mat_sub(h_hat, h_t, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_h_t, tmp, grad_z_t, hidden_unit, batch_size);

    // d loss / d h_t_1
    mat_one_sub(z_t, tmp, hidden_unit, batch_size);
    mat_add(grad_h_t, tmp, grad_h_t_1, hidden_unit, batch_size);

    // d loss / d h_hat
    mat_hadamard(grad_h_t, z_t, grad_h_hat, hidden_unit, batch_size);

    // d loss / d h_hat_before_sigmoid
    mat_one_sub(grad_h_hat, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_h_hat, tmp, grad_h_hat_before_sigmoid, hidden_unit, batch_size);

    // d loss / Wh 
    // tmp2: transpose of x_t
    float* tmp2 = mat_transpose(x_t, vec_len, batch_size);
    mat_multiplication(tmp2, grad_h_hat_before_sigmoid, grad_w_h, hidden_unit, vec_len, batch_size);

    // d loss / u_h
    mat_hadamard(r_t, h_t_1, tmp, hidden_unit, batch_size);
    float *tmp3 = mat_transpose(tmp, hidden_unit, batch_size);
    mat_multiplication(tmp3, grad_h_hat_before_sigmoid, grad_u_h, hidden_unit, hidden_unit, batch_size);

    // d loss / b_h
    sum_over_rows(grad_h_hat_before_sigmoid, grad_b_h, hidden_unit, batch_size);

    // d loss / r_t
    float *tmp4 = mat_transpose(u_h, hidden_unit, hidden_unit);
    mat_multiplication(grad_h_hat_before_sigmoid, tmp4, tmp, hidden_unit, batch_size, hidden_unit);
    mat_div(tmp, h_t_1, grad_r_t, hidden_unit, batch_size);

    // d loss / h_t
    mat_div(tmp, r_t, tmp, hidden_unit, batch_size);
    mat_add(grad_h_t_1, tmp, grad_h_t_1, hidden_unit, batch_size);

    // d loss / d r_t_before_sigmoid
    mat_one_sub(grad_r_t, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_r_t, tmp, grad_r_t_before_sigmoid, hidden_unit, batch_size);

    // d loss / d w_r
    mat_multiplication(tmp2, grad_r_t_before_sigmoid, grad_w_r, hidden_unit, vec_len, batch_size);
    //Print(grad_w_r, hidden_unit, vec_len);

    // d loss / d u_r
    free(tmp3);
    tmp3 = mat_transpose(h_t_1, hidden_unit, batch_size);
    mat_multiplication(tmp3, grad_r_t_before_sigmoid, grad_u_r, hidden_unit, hidden_unit, batch_size);

    // d loss / d b_r
    sum_over_rows(grad_r_t_before_sigmoid, grad_b_r, hidden_unit, batch_size);

    // d loss / d h_t
    free(tmp4);
    tmp4 = mat_transpose(u_r, hidden_unit, hidden_unit);
    mat_multiplication(grad_r_t_before_sigmoid, tmp4, tmp, hidden_unit, batch_size, hidden_unit);
    // if (i_start == 0) Print(grad_r_t_before_sigmoid, hidden_unit * batch_size);

    mat_add(grad_h_t_1, tmp, grad_h_t_1, hidden_unit, batch_size);
    // if(i_start == 0) Print(grad_h_t_1, hidden_unit * batch_size);

    // d loss / d z_t_before_sigmoid
    mat_one_sub(grad_z_t, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_z_t, tmp, grad_z_t_before_sigmoid, hidden_unit, batch_size);
    //Print(grad_z_t_before_sigmoid, hidden_unit, batch_size);

    // d loss / d w_z
    mat_multiplication(tmp2, grad_z_t_before_sigmoid, grad_w_z, hidden_unit, vec_len, batch_size);
    //Print(grad_w_z, hidden_unit, vec_len);

    // d loss / d u_z
    mat_multiplication(tmp3, grad_z_t_before_sigmoid, grad_u_z, hidden_unit, hidden_unit, batch_size);
    //Print(grad_u_z, hidden_unit, hidden_unit);

    // d loss / d b_z
    sum_over_rows(grad_z_t_before_sigmoid, grad_b_z, hidden_unit, batch_size);
    //Print(grad_b_z, hidden_unit, 1);

    // d loss / d h_t;
    free(tmp4);
    tmp4 = mat_transpose(u_z, hidden_unit, hidden_unit);
    mat_multiplication(grad_z_t_before_sigmoid, tmp4, tmp, hidden_unit, batch_size, hidden_unit);
    mat_add(grad_h_t_1, tmp, grad_h_t_1, hidden_unit, batch_size);
    //Print(grad_h_t_1, hidden_unit, batch_size);

    // loss for next timestep
    memcpy(grad_h_t_1, grad_h_t, batch_size * hidden_unit * sizeof(float));
    //Print(grad_h_t, batch_size, hidden_unit);

    // cumulate gradient for all timesteps;

    update_variable(w_z, grad_w_z, hidden_unit, vec_len, step_size);
    update_variable(w_r, grad_w_r, hidden_unit, vec_len, step_size);
    update_variable(w_h, grad_w_h, hidden_unit, vec_len, step_size);

    mat_add(Grad_u_z, grad_u_z, Grad_u_z, hidden_unit, hidden_unit);
    mat_add(Grad_u_r, grad_u_r, Grad_u_r, hidden_unit, hidden_unit);
    mat_add(Grad_u_h, grad_u_h, Grad_u_h, hidden_unit, hidden_unit);

    update_variable(b_z, grad_b_z, hidden_unit, 1, step_size);
    update_variable(b_r, grad_b_r, hidden_unit, 1, step_size);
    update_variable(b_h, grad_b_h, hidden_unit, 1, step_size);

    free(z_t);
    free(r_t);
    free(h_hat);
    free(h_t_1);

    free(grad_w_z);
    free(grad_w_r);
    free(grad_w_h);
    free(grad_u_z);
    free(grad_u_r);
    free(grad_u_h);
    free(grad_b_z);
    free(grad_b_r);
    free(grad_b_h); 
    free(grad_r_t);
    free(grad_r_t_before_sigmoid);
    free(grad_z_t);
    free(grad_z_t_before_sigmoid);
    free(grad_h_hat);
    free(grad_h_hat_before_sigmoid);
    free(grad_h_t_1);

}

void run_model(int num_data, int batch_size, int window_size, int vec_len, int hidden_unit, float step_size,
                        float* h_t, float* h_t_new,
                        float* w_z, float* w_r, float* w_h,
                        float* u_z, float* u_r, float* u_h,
                        float* b_z, float* b_r, float* b_h,
                        float* dense, float* predict, float* arr_data, int m, int n, float* y, int iter) {
    
    double startTime = CycleTimer::currentSeconds();
    for (int num_iter = 0; num_iter < iter; num_iter++) {
        printf("begin iter %d\n", num_iter);
        // One iteration, loop through all data point
        for (int i = 0; i < num_data; i += batch_size) {

            // batch_size * (num_data * vec_len)
            int start_i = i;
            int end_i = min(num_data, i + batch_size);
            int batch = end_i - start_i;

            // reset gradients
            float *Z = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
            float *R = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
            float *H_hat = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
            float *H_1 = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
            float* x_t = (float*)calloc(batch_size * vec_len, sizeof(float));

            float* grad_h_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
            float* Grad_u_z = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
            float* Grad_u_r = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
            float* Grad_u_h = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));

            // for each time step
            for (int j = 0; j < window_size; j++) {

                // Construct x_t
                for (int _m = 0; _m < batch; _m++) {
                    for (int _n = 0; _n < vec_len; _n++) {
                        x_t[_m * vec_len + _n] = arr_data[(start_i + _m) * n + j + _n];
                    }
                }

                // one forward iteration: 
                gru_forward(j, Z, H_hat, H_1, R, 
                    batch_size, vec_len, hidden_unit, x_t, h_t, h_t_new, 
                    w_z, w_r, w_h, u_z, u_r, u_h, b_z, b_r, b_h); 

                // update h_t for next round
                memcpy(h_t, h_t_new, batch_size * hidden_unit * sizeof(float));
                memset(h_t_new, 0.f, batch_size * hidden_unit * sizeof(float));
            }

            // inference
            mat_multiplication(h_t, dense, predict, 1, batch_size, hidden_unit);

            // calculate loss
            float loss = calculate_loss(batch, &y[start_i], predict);
            cout << "loss is " << loss << endl;        

            // reset gradients
            memset(grad_h_t, 0.f, batch_size * hidden_unit * sizeof(float));
            
            // calculate gradients for predice, dense, and h_t
            update_dense_and_grad_h_t(batch, hidden_unit, batch_size, step_size, loss,
                                    dense, grad_h_t, predict, h_t, &y[start_i]);

            // calculate gradient for each time_step
            for (int j = window_size-1; j >= 1; j--) {

                // Construct x_t
                for (int _m = 0; _m < batch; _m++) {
                    for (int _n = 0; _n < vec_len; _n++) {
                        x_t[_m * vec_len + _n] = arr_data[(start_i + _m) * n + j + _n];
                    }
                }

                if (j != window_size - 1) {

                    for (int i = 0; i < batch_size * hidden_unit; i++) {
                        h_t[i] = H_1[(j+1)*batch_size*hidden_unit];
                    }

                }

                gru_backward(i, vec_len, hidden_unit, batch_size, step_size,
                            grad_h_t, h_t,
                            x_t, 
                            u_z, u_r, u_h, w_z, w_r, w_h, 
                            b_z, b_r, b_h, Grad_u_z, Grad_u_r, Grad_u_h,
                            &Z[j*hidden_unit*batch_size],
                            &R[j*hidden_unit*batch_size],
                            &H_hat[j*hidden_unit*batch_size],
                            &H_1[j*hidden_unit*batch_size]);
            }

            // update variables
            update_variable(u_z, Grad_u_z, hidden_unit, hidden_unit, step_size);
            update_variable(u_r, Grad_u_r, hidden_unit, hidden_unit, step_size);
            update_variable(u_h, Grad_u_h, hidden_unit, hidden_unit, step_size);

            free(Z);
            free(R);
            free(H_hat);
            free(H_1);
            free(x_t);

            free(grad_h_t);
            free(Grad_u_z);
            free(Grad_u_r);
            free(Grad_u_h);
        }
    }


    double endTime = CycleTimer::currentSeconds();
    printf("CPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));
}

#endif // _GRU_SEQUANTIAL_H_