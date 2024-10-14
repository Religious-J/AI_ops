#include <math.h>
#include<iostream>


static float silu(float x) {
    return x / (1.0f + expf(-x));
}

void Group_norm_Silu(float* input, float* output, int N, int H, int W, int C, int G, float epsilon, float* gamma, float* beta) {
    int group_size = C / G;

    // Iterate over batch
    for (int n = 0; n < N; ++n) {
        // Iterate over groups
        for (int g = 0; g < G; ++g) {
            // Calculate mean and variance for the group
            float mean = 0.0;
            float var = 0.0;

            // Accumulate mean and variance for this group
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
                        int index = n * H * W * C + h * W * C + w * C + c;
                        mean += input[index];
                    }
                }
            }
            mean /= (H * W * group_size);

            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
                        int index = n * H * W * C + h * W * C + w * C + c;
                        var += (input[index] - mean) * (input[index] - mean);
                    }
                }
            }
            var /= (H * W * group_size);
            var = sqrt(var + epsilon);
            // Normalize and apply gamma and beta
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = g * group_size; c < (g + 1) * group_size; ++c) {
                        int index = n * H * W * C + h * W * C + w * C + c;
                        output[index] = (input[index] - mean) / var;
                        if (gamma != NULL) {
                            output[index] *= gamma[c];
                        }
                        if (beta != NULL) {
                            output[index] += beta[c];
                        }
                        output[index] = silu(output[index]);
                    }
                }
            }
        }
    }
}


int main(){
    const size_t N = 1; 
    const size_t H = 2;
    const size_t W = 2;
    const size_t C = 4;

    const size_t P = 16;
    const size_t Q = 16;
    const size_t K = 80;
    const size_t R = 3;
    const size_t S = 3;

    const size_t group_norm_group_count = 2;
    double eps = 0.00001;

    int input_num = N*H*W*C;
    int output_num = N*H*W*K;

    float host_input_tensor_ft32[100] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,13.0, 14.0, 15.0, 16.0};
    
    float host_gamma_ft32[10] = {1.0, 1.0, 1.0, 1.0};
    float host_beta_ft32[10] = {0.0, 0.0, 0.0, 0.0};

    float* host_output_tensor_ft32 = (float*)malloc(input_num * sizeof(float));
    float* host_result_tensor_ft32 = (float*)malloc(output_num * sizeof(float));
    float* host_conv_weight_ft32 = (float*)malloc(C*K*H*W * sizeof(float));
    float* host_conv_bias_ft32 = (float*)malloc(K * sizeof(float));


    Group_norm_Silu(host_input_tensor_ft32, host_output_tensor_ft32, N, H ,W, C, group_norm_group_count, eps, host_gamma_ft32, host_beta_ft32);

    for(int i=0; i<16; i++) {
        printf("%f ", host_output_tensor_ft32[i]);
    }
    std::cout << std::endl;



    // free(host_input_tensor_ft32);
    // free(host_gamma_ft32);
    // free(host_beta_ft32);
    free(host_output_tensor_ft32);
    free(host_result_tensor_ft32);
    free(host_conv_weight_ft32);
    free(host_conv_bias_ft32);
}