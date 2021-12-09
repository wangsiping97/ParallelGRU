#include <fstream>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <stdio.h>
#include <cstdlib>
#include <stdio.h>
#include <getopt.h>
#include "gru_sequential.h"

using namespace std;


void print_cuda_info();
void run_model_cuda(int num_data, int batch_size, int window_size, int vec_len, int hidden_unit, float step_size, 
                        float* old_h_t, float* new_h_t,
                        float* w_z, float* w_r, float* w_h,
                        float* u_z, float* u_r, float* u_h,
                        float* b_z, float* b_r, float* b_h,
                        float* dense, float* predict, float* data, int m, int n, float* y, int iter);

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
        printf("%.6f ", a[i]);
    }
    cout << endl;
}

int main(int argc, char** argv) {
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"iter", 1, 0, 'i'},
        {"gpu",  1, 0, 'g'},
        {"help", 0, 0, '?'},
        {0 ,0, 0, 0}
    };

    bool use_gpu = false;
    int iter = 1;

    while ((opt = getopt_long(argc, argv, "i:g:?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'g':
            use_gpu = (bool)(atoi(optarg));
            break;
        case 'i':
            iter = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }

    // read input
    fstream infile("data/data_sliding.csv");
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

    fstream label("data/y.csv");

    while (label >> line) {

        stringstream s(line);
   
        for (float i; s >> i;) {
            y.push_back(i);

            if (s.peek() == ',')
                s.ignore();
        }
    }

    int num_data = 3000;
    int window_size = 20;
    int vec_len = 280;

    int batch_size = 500;

    int hidden_unit = 100;
    float step_size = 0.00001;

    // allocate variables
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

    // using GPU
    if (use_gpu) {
        print_cuda_info();
        run_model_cuda(num_data, batch_size, window_size, vec_len, hidden_unit, step_size, 
                            h_t, h_t_new, 
                            w_z, w_r, w_h,
                            u_z, u_r, u_h,
                            b_z, b_r, b_h,
                            dense, predict, arr_data, m , n, &y[0], iter);
        return 0;
    } 

    // using CPU
    else {
        cout << "Using CPU..." << endl;
        run_model(num_data, batch_size, window_size, vec_len, hidden_unit, step_size,
                        h_t, h_t_new,
                        w_z, w_r, w_h,
                        u_z, u_r, u_h,
                        b_z, b_r, b_h,
                        dense, predict, arr_data, m, n, &y[0], iter);
        return 0;
    }

}