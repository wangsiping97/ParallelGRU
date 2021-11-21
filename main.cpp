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

double gru_forward_cuda(int batch_size, int x_width, int hidden_unit,
                float* x_t, float* old_h_t, float* new_h_t,
                float* w_z, float* w_r, float* w_h,
                float* u_z, float* u_r, float* u_h,
                float* b_z, float* b_r, float* b_h);
void print_cuda_info();

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
        weight[i] = random_num();
    }
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

    if (use_gpu) {
        print_cuda_info();
        double startTime = CycleTimer::currentSeconds();
        double kernalTime = 0;
        // One iteration, loop through all data point
        for (int i = 0; i < num_data; i += batch_size) {

            // batch_size * (num_data * vec_len)
            int start_i = i;
            int end_i = min(num_data, i + batch_size);
            int batch = end_i - start_i;

            // for each time step
            for (int j = 0; j < window_size; j++) {

                // Construct x_t
                for (int m = 0; m < batch; m++) {
                    for (int n = 0; n < vec_len; n++) {
                        x_t[m * vec_len + n] = data[start_i + m][j + n];
                    }
                }

                // one forward iteration: 
                kernalTime += gru_forward_cuda(batch_size, vec_len, hidden_unit, x_t, h_t, h_t_new, 
                    w_z, w_r, w_h, u_z, u_r, u_h, b_z, b_r, b_h); 
            
                // for (int k = 0; k < batch_size * hidden_unit; k++)
                //     cout << h_t_new[k] << endl;
                
                memcpy(h_t, h_t_new, batch_size * hidden_unit * sizeof(float));
                memset(h_t_new, 0.f, batch_size * hidden_unit * sizeof(float));
            }

            // inference
            mat_multiplication(dense, h_t, predict, batch_size, 1, hidden_unit);
            // for (int k = 0; k < batch_size; k++)
            //     cout << predict[k] << endl;
            if (i == 0) {
                for (int k = 0; k < batch_size; k++) {
                    cout << predict[k] << " ";
                }
                cout << endl;
            }
            
            // calculate loss
            // gru_backward
            // update variables
            
        }
        double endTime = CycleTimer::currentSeconds();
        printf("GPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));
        printf("GPU Kernel: %.3f ms\n", 1000.f * kernalTime);
    } else {
        cout << "Using CPU..." << endl;
        double startTime = CycleTimer::currentSeconds();
        // One iteration, loop through all data point
        for (int i = 0; i < num_data; i += batch_size) {

            // batch_size * (num_data * vec_len)
            int start_i = i;
            int end_i = min(num_data, i + batch_size);
            int batch = end_i - start_i;

            // for each time step
            for (int j = 0; j < window_size; j++) {

                // Construct x_t
                for (int m = 0; m < batch; m++) {
                    for (int n = 0; n < vec_len; n++) {
                        x_t[m * vec_len + n] = data[start_i + m][j + n];
                    }
                }

                // one forward iteration: 
                gru_forward(batch_size, vec_len, hidden_unit, x_t, h_t, h_t_new, 
                    w_z, w_r, w_h, u_z, u_r, u_h, b_z, b_r, b_h); 
            
                // for (int k = 0; k < batch_size * hidden_unit; k++)
                //     cout << h_t_new[k] << endl;
                
                memcpy(h_t, h_t_new, batch_size * hidden_unit * sizeof(float));
                memset(h_t_new, 0.f, batch_size * hidden_unit * sizeof(float));
            }

            // inference
            mat_multiplication(dense, h_t, predict, batch_size, 1, hidden_unit);
            if (i == 0) {
                for (int k = 0; k < batch_size; k++) {
                    cout << predict[k] << " ";
                }
                cout << endl;
            }
                    
            
            // calculate loss
            // gru_backward
            // update variables
            
        }
        double endTime = CycleTimer::currentSeconds();
        printf("CPU Overall: %.3f ms\n", 1000.f * (endTime - startTime));
    }
}