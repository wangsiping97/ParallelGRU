#ifndef _GRU_SEQUANTIAL_H_
#define _GRU_SEQUANTIAL_H_

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

void mat_hadamard_no_overwrite(float*a, float* b, float* res, int width, int height) {
    for (int i = 0; i < width * height; ++i) {
        res[i] += a[i] * b[i];
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
/*
    for (int i = timestep*hidden_unit*batch_size; i < (timestep+1) *hidden_unit * batch_size; i++) {
        std::cout << R[i] << " ";
    }
    std::cout << std::endl;
*/

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

void gru_backward(int vec_len, int hidden_unit, int batch_size, float step_size,
                  float *grad_h_t, float *h_hat, float *h_t_1, float *h_t,
                  float *x_t, float *z_t, float *r_t,
                  float *u_h, float *u_r, float *u_z, 
                  float *w_z, float *w_r, float *w_h, 
                  float *b_z, float *b_r, float *b_h,
                  float *Grad_u_z, float *Grad_u_r, float *Grad_u_h) {
    
    // reset gradients for current timestep
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
    //Print(grad_z_t, hidden_unit, batch_size);

    // d loss / d h_t_1
    mat_one_sub(z_t, tmp, hidden_unit, batch_size);
    mat_add(grad_h_t, tmp, grad_h_t_1, hidden_unit, batch_size);
    //Print(grad_h_t_1, hidden_unit, batch_size);

    // d loss / d h_hat
    mat_hadamard(grad_h_t, z_t, grad_h_hat, hidden_unit, batch_size);
    //Print(grad_h_hat, hidden_unit, batch_size);

    // d loss / d h_hat_before_sigmoid
    mat_one_sub(grad_h_hat, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_h_hat, tmp, grad_h_hat_before_sigmoid, hidden_unit, batch_size);
    //Print(grad_h_hat_before_sigmoid, hidden_unit, batch_size);

    // d loss / Wh 
    // tmp2: transpose of x_t
    float* tmp2 = mat_transpose(x_t, vec_len, batch_size);
    mat_multiplication(tmp2, grad_h_hat_before_sigmoid, grad_w_h, hidden_unit, vec_len, batch_size);
    //Print(grad_w_h, hidden_unit, vec_len);

    // d loss / u_h
    mat_hadamard(r_t, h_t_1, tmp, hidden_unit, batch_size);
    float *tmp3 = mat_transpose(tmp, hidden_unit, batch_size);
    mat_multiplication(tmp3, grad_h_hat_before_sigmoid, grad_u_h, hidden_unit, hidden_unit, batch_size);
    //Print(grad_u_h, hidden_unit, hidden_unit);

    // d loss / b_h
    sum_over_rows(grad_h_hat_before_sigmoid, grad_b_h, hidden_unit, batch_size);
    //Print(grad_b_h, hidden_unit, 1);

    // d loss / r_t
    float *tmp4 = mat_transpose(u_h, hidden_unit, hidden_unit);
    mat_multiplication(grad_h_hat_before_sigmoid, tmp4, tmp, hidden_unit, batch_size, hidden_unit);
    mat_div(tmp, h_t_1, grad_r_t, hidden_unit, batch_size);
    //Print(grad_r_t, hidden_unit, batch_size);

    // d loss / h_t
    mat_div(tmp, r_t, tmp, hidden_unit, batch_size);
    mat_add(grad_h_t_1, tmp, grad_h_t_1, hidden_unit, batch_size);
    //Print(grad_h_t_1, hidden_unit, batch_size);

    // d loss / d r_t_before_sigmoid
    mat_one_sub(grad_r_t, tmp, hidden_unit, batch_size);
    mat_hadamard(grad_r_t, tmp, grad_r_t_before_sigmoid, hidden_unit, batch_size);
    //Print(grad_r_t_before_sigmoid, hidden_unit, batch_size);

    // d loss / d w_r
    mat_multiplication(tmp2, grad_r_t_before_sigmoid, grad_w_r, hidden_unit, vec_len, batch_size);
    //Print(grad_w_r, hidden_unit, vec_len);

    // d loss / d u_r
    free(tmp3);
    tmp3 = mat_transpose(h_t_1, hidden_unit, batch_size);
    mat_multiplication(tmp3, grad_r_t_before_sigmoid, grad_u_r, hidden_unit, hidden_unit, batch_size);
    //Print(grad_u_r, hidden_unit, hidden_unit);

    // d loss / d b_r
    sum_over_rows(grad_r_t_before_sigmoid, grad_b_r, hidden_unit, batch_size);
    //Print(grad_b_r, hidden_unit, 1);

    // d loss / d h_t
    free(tmp4);
    tmp4 = mat_transpose(u_r, hidden_unit, hidden_unit);
    mat_multiplication(grad_r_t_before_sigmoid, tmp4, tmp, hidden_unit, batch_size, hidden_unit);

    mat_add(grad_h_t_1, tmp, grad_h_t_1, hidden_unit, batch_size);
    //Print(grad_h_t_1, hidden_unit, batch_size);

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

}

#endif // _GRU_SEQUANTIAL_H_