#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cmath>
#include <cstring>
#include <stdio.h>
#include <cstdlib>
#include <stdio.h>
#include <getopt.h>
#include "gru_sequential.h"
#include "CycleTimer.h"

using namespace std;


void print_cuda_info();
void one_iteration_cuda(int num_data, int batch_size, int window_size, int vec_len, int hidden_unit,
                        float* old_h_t, float* new_h_t,
                        float* w_z, float* w_r, float* w_h,
                        float* u_z, float* u_r, float* u_h,
                        float* b_z, float* b_r, float* b_h,
                        float* dense, float* predict, float* data, int m, int n);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Program Options:\n");
    printf("  -g  --gpu <BOOL>  whether to use GPU\n");
    printf("  -?  --help             This message\n");
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
        weight[i] = random_num() / 1000;
    }
}

void Print(float *a, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        cout << a[i] << " ";
    }
    cout << endl;
}

int main(int argc, char** argv) {
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"gpu",  1, 0, 'g'},
        {"help", 0, 0, '?'},
        {0 ,0, 0, 0}
    };

    bool use_gpu = false;

    while ((opt = getopt_long(argc, argv, "g:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'g':
            use_gpu = (bool)(atoi(optarg));
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

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
    // cout << data[0][0] << endl;
    // cout << data.size() << endl;
    
    // copy data into an array
    int m = data.size();
    int n = data[0].size();
    float* arr_data = (float*)calloc(m * n, sizeof(float));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            arr_data[i * n + j] = data[i][j];
        }
    }

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

    // cout << y.size() << endl;
    // cout << y[0] << endl;


    int num_data = 3000;
    int window_size = 20;
    int vec_len = 280;

    int batch_size = 500;

    int hidden_unit = 100;
    float step_size = 0.00001;

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

    float *Z = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
    float *R = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
    float *H_hat = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));
    float *H_1 = (float*)calloc(window_size * batch_size * hidden_unit, sizeof(float));

    float* dense = (float*)calloc(hidden_unit * 1, sizeof(float));
    float* predict = (float*)calloc(batch_size * 1, sizeof(float));

    float* grad_h_t = (float*)calloc(batch_size * hidden_unit, sizeof(float));
    float* Grad_u_z = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* Grad_u_r = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));
    float* Grad_u_h = (float*)calloc(hidden_unit * hidden_unit, sizeof(float));

    // initialize variables
    init(w_z, vec_len * hidden_unit);
    init(w_r, vec_len * hidden_unit);
    init(w_h, vec_len * hidden_unit);
    init(u_z, hidden_unit * hidden_unit);
    init(u_r, hidden_unit * hidden_unit);
    init(u_h, hidden_unit * hidden_unit);
    init(dense, hidden_unit);

    // using GPU
    if (use_gpu) {
        print_cuda_info();
        one_iteration_cuda(num_data, batch_size, window_size, vec_len, hidden_unit,
                            h_t, h_t_new, 
                            w_z, w_r, w_h,
                            u_z, u_r, u_h,
                            b_z, b_r, b_h,
                            dense, predict, arr_data, m , n);
        return 0;
    } 

    // using CPU
    cout << "Using CPU..." << endl;
    double startTime = CycleTimer::currentSeconds();
    // One iteration, loop through all data point
    for (int i = 0; i < num_data; i += batch_size) {

        // batch_size * (num_data * vec_len)
        int start_i = i;
        int end_i = min(num_data, i + batch_size);
        int batch = end_i - start_i;

        // reset gradients

        memset(Grad_u_z, 0.f, hidden_unit * hidden_unit * sizeof(float));
        memset(Grad_u_r, 0.f, hidden_unit * hidden_unit * sizeof(float));
        memset(Grad_u_h, 0.f, hidden_unit * hidden_unit * sizeof(float));

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

                //Print(h_t, batch_size, hidden_unit);
            }

            gru_backward(vec_len, hidden_unit, batch_size, step_size,
                        grad_h_t, h_t,
                        x_t, 
                        u_h, u_r, u_z, w_z, w_r, w_h, 
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

        //Print(w_r, hidden_unit, 1);
    }


    double endTime = CycleTimer::currentSeconds();
    printf("CPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));

}