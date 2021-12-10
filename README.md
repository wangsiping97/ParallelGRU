# ParallelGRU
A parallel version of GRU (Gated Recurrent Unit) model using OpenMP and CUDA, by Siping Wang and Junyan Pu. Final project for CMU 15-618. 

## Project Page
[https://wangsiping97.github.io/15618/](https://wangsiping97.github.io/15618/)

## Usage

**Note:** 
1. The NVIDIA CUDA C?C++ Compiler (NVCC) needs to be added to `PATH`. 
2. The CUDA shared library must be loaded at runtime. 

```
$ cd src
$ make
$ ./cudaGRU -g <1 for GPU or 0 for CPU> -i <number of iterations>
```