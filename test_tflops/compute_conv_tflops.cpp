#include <iostream>

// pad = 0
// stride = 1
// dilation = 1

// x => [N*P*Q, C*R*S]
// w => [C*R*S, K]

void compute_flops(int N, int P, int Q, int C, int R, int S, int K, double time){
    long long int flops = (long long int) N * P * Q * C * R * S * K * 2;
    double value = 1e-6;
    double ans = flops * value;
    double res = ans / time;
    std::cout << res << std::endl;
}


int main(){

    int N = 2;
    int H[100] = {64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 16, 16, 64, 32, 16};
    int W[100] = {64, 64, 64, 32, 32, 32, 32, 32, 16, 16, 16, 16, 64, 32, 16};
    int C[100] = {320, 640, 960, 320, 640, 960, 1280, 1920, 640, 1280, 1920, 2560, 80, 160, 320};
    int K[100] = {80, 80, 80, 160, 160, 160, 160, 160, 320, 320, 320, 320, 320, 640, 1280};

    double time[100] = {208.153,    // us
                      690.012,
                      1040.807,
                      106.508,
                      272.151,
                      665.884,
                      551.685,
                      779.179,
                      250.771,
                      447.015,
                      640.239,
                      837.061,
                      145.378,
                      148.915,
                      475.002};
    int num = 15;
    for(int i=0; i<15; i++){
        compute_flops(N, H[i], W[i], C[i], 3, 3, K[i], time[i]);
    }

    return 0;
}