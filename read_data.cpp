#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <cstdlib>
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

    // initialize new_h_t
    memset(new_h_t, 0, sizeof(new_h_t));

    float* tmp1 = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    float* tmp2 = (float*)malloc(hidden_unit * batch_size * sizeof(float));

    // z_t = sigmoid(x_t * w_z + u_z * old_h_t + b_z)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(x_t, w_z, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_z, old_h_t, tmp2, hidden_unit, batch_size, hidden_unit);
    float* z_t = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, z_t, hidden_unit, batch_size);
    mat_add_b(z_t, b_z, z_t, hidden_unit, batch_size);
    mat_sigmoid(z_t, hidden_unit, batch_size);

    // r_t = sigmoid(x_t * w_r + u_r * old_h_t + b_r)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(x_t, w_r, tmp1, hidden_unit, batch_size, x_width);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_r, old_h_t, tmp2, hidden_unit, batch_size, hidden_unit);
    float* r_t = (float*)malloc(hidden_unit * batch_size * sizeof(float));
    mat_add(tmp1, tmp2, r_t, hidden_unit, batch_size);
    mat_add_b(r_t, b_r, r_t, hidden_unit, batch_size);
    mat_sigmoid(r_t, hidden_unit, batch_size);

    // h_hat = phi(x_t * w_h + u_h(r_t . old_h_t) + b_h)
    memset(tmp1, 0, sizeof(tmp1));
    mat_multiplication(x_t, w_h, tmp1, hidden_unit, batch_size, x_width);
    mat_hadamard(r_t, old_h_t, r_t, hidden_unit, batch_size);
    memset(tmp2, 0, sizeof(tmp2));
    mat_multiplication(u_h, r_t, tmp2, hidden_unit, batch_size, hidden_unit);
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

    // free temp arrays
    free(tmp1);
    free(tmp2);
    free(tmp3);
    free(z_t);
    free(r_t);
    free(h_hat);
}

// random number betwee 0 and 1
float random_num() {
    int min = -1.f;
    int max = 1.f;

    float r = (float)rand() / (float)RAND_MAX;
    return min + r * (max - min);
}

void init(float *weight, size_t size) {

    for (size_t i = 0; i < size; i++) {
        weight[i] = random_num();
    }
}

int main() {

    // read input
    fstream infile("data_sliding.csv");
    string line, word;
    
    vector<vector<float> > data;

    while (infile >> line) {

        vector<float> row;
        stringstream s(line);
   
        for (float i; s >> i;) {
            row.push_back(i);

            if (s.peek() == ',')
                s.ignore();
        }
        
        data.push_back(row);
    
    }

    // read label
    cout << data[0][0] << endl;
    cout << data.size() << endl;

    vector<float> y;

    fstream label("y.csv");

    while (label >> line) {

        stringstream s(line);
   
        for (float i; s >> i;) {
            y.push_back(i);

            if (s.peek() == ',')
                s.ignore();
        }
    }

    cout << y.size() << endl;
    cout << y[0] << endl;


    int num_data = 3000;
    int window_size = 20;
    int vec_len = 28;

    int batch_size = 50;

    int hidden_unit = 10;

    // allocate variables
    float* x_t = (float*)calloc(batch_size * vec_len, sizeof(float));
    float* h_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* h_t_new = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* w_z = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* w_r = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* w_h = (float*)calloc(vec_len * hidden_unit, sizeof(float));
    float* u_z = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* u_r = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* u_h = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* b_z = (float*)calloc(hidden_unit, sizeof(float));
    float* b_r = (float*)calloc(hidden_unit, sizeof(float));
    float* b_h = (float*)calloc(hidden_unit, sizeof(float));

    float* dense = (float*)calloc(hidden_unit * 1, sizeof(float));
    float* predict = (float*)calloc(batch_size * 1, sizeof(float));

    // initialize variables
    init(w_z, vec_len * hidden_unit);
    init(w_r, vec_len * hidden_unit);
    init(w_h, vec_len * hidden_unit);
    init(u_z, hidden_unit * hidden_unit);
    init(u_r, hidden_unit * hidden_unit);
    init(u_h, hidden_unit * hidden_unit);
    init(dense, hidden_unit);

    // One iteration, loop through all data point
    for (int i = 0; i < num_data; i += batch_size) {

        // batch_size * (num_data * vec_len)
        int start_i = i;
        int end_i = min(num_data, i + batch_size);
        int batch = end_i - start_i;

        // for each time step
        for (int j = 0; j < window_size; j++) {

            int start_j = j * vec_len;
            int end_j = start_j + vec_len;

            // Construct x_t
            for (int m = 0; m < batch; m++) {
                for (int n = 0; n < vec_len; n++) {
                    x_t[m * vec_len + n] = data[start_i + m][j + n];
                }
            }

            // one forward iteration
            gru_forward(batch_size, vec_len, hidden_unit, x_t, h_t, h_t_new, 
                w_z, w_r, w_h, u_z, u_r, u_h, b_z, b_r, b_h); 
           
            for (int k = 0; k < batch_size * hidden_unit; k++)
                cout << h_t_new[k] << endl;
            
            h_t = h_t_new;
            memset(h_t_new, 0.f, batch_size * hidden_unit * sizeof(float));
        }
        
        // inference
        mat_multiplication(h_t, dense, predict, batch_size, 1, hidden_unit);
        for (int k = 0; k < batch_size; k++)
            cout << predict[k] << endl;
    }
}