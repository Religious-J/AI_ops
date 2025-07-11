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

void conv2d(
    float *input,        // (N, H, W, C)
    float *kernel,       // (C, R, S, K)
    float *bias,         // (K)
    float *output,       // (N, H, W, K)
    int N, int H, int W, int C, int K, int R, int S,
    int padding,
    int stride
) {
    int H_out = (H + 2 * padding - R) / stride + 1;
    int W_out = (W + 2 * padding - S) / stride + 1;

    for (int n = 0; n < N; n++) {
        for (int c_out = 0; c_out < K; c_out++) {
            for (int h = 0; h < H_out; h++) {
                for (int w = 0; w < W_out; w++) {
                    float sum = 0.0f;
                    for (int c_in = 0; c_in < C; c_in++) {
                        for (int i = 0; i < R; i++) {
                            for (int j = 0; j < S; j++) {
                                int h_in = h * stride - padding + i;
                                int w_in = w * stride - padding + j;

                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    sum += input[n * H * W * C + h_in * W * C + w_in * C + c_in] *
                                           kernel[c_in * R * S * K + i * S * K + j * K + c_out];
                                }
                            }
                        }
                    }
                    if ( bias !=  nullptr) {
                        sum += bias[c_out];
                    }
                    output[n * H_out * W_out * K + h * W_out * K + w * K + c_out] = sum;
                }
            }
        }
    }
}

int main(){
    const size_t N = 1; 
    const size_t H = 3;
    const size_t W = 3;
    const size_t C = 2;

    const size_t P = 2;
    const size_t Q = 2;
    const size_t K = 2;
    const size_t R = 2;
    const size_t S = 2;

    const size_t group_norm_group_count = 2;
    double eps = 0.00001;

    int input_num = N*H*W*C;
    int output_num = N*H*W*K;

    float host_input_tensor_ft32[100] = {-0.8047, 0.321, -0.02548, 0.6445, -0.3008, 0.3894, -0.1074, -0.48, 0.595, -0.4646, 0.6675, -0.806, -1.196, -0.406, -0.1824, 0.1032};
    // float host_input_tensor_ft32[100] = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    float host_gamma_ft32[10] = {1.0, 1.0, 1.0, 1.0};
    float host_beta_ft32[10] = {0.0, 0.0, 0.0, 0.0};

    float* host_output_tensor_ft32 = (float*)malloc(input_num * sizeof(float));
    float* host_result_tensor_ft32 = (float*)malloc(output_num * sizeof(float));

    float host_conv_weight_ft32[2 * 3 * 3 * 4] = {
        -0.3347, -0.0995, 0.4072, 0.9194,
         0.312,  1.533, -0.5503, -0.383,
        -0.8228,  1.6, -0.0693,  0.0832,
        -0.327, -0.0458, -0.3044,  1.923,
        -0.0787, -0.582, -1.618,  0.867,
        -1.04,   0.6504,  2.7,   0.8022,
        -1.097, -0.1781, -0.4229, -0.3303,
        -1.111, -0.742,  2.574,  1.073,
        // -1.866, -0.647,  1.082,  0.1766,
        // -0.8354, -1.695,  1.134,  1.049,
        // -2.129, -1.4375,  0.178,  1.395,
        //  0.2913, -0.08203, 0.644,  0.3281,
        //  0.8574, -0.937,  0.18,  -1.424,
        // -0.3677, -1.523, -0.635,  0.9873,
        // -1.016,  2.045,  0.25,   0.6514,
        // -1.266,  1.374, -0.61,   0.03076,
        //  0.82,   1.454, -0.5835, 0.4153,
        //  0.667,  0.8696, -1.203,  2.861
    };

    // float host_conv_weight_ft32[100] = {1,0.1,0.2,-1,1,0,0,-1,0,1,-1,0,0,1,-1,0};

    float host_conv_bias_ft32[2] = {0, 0};


    Group_norm_Silu(host_input_tensor_ft32, host_output_tensor_ft32, N, H ,W, C, group_norm_group_count, eps, host_gamma_ft32, host_beta_ft32);

    for(int i=0; i<16; i++) {
        printf("%f ", host_output_tensor_ft32[i]);
    }
    std::cout << std::endl;

    conv2d(host_output_tensor_ft32, host_conv_weight_ft32, host_conv_bias_ft32, host_result_tensor_ft32, N, H, W, C, K, R, S, 0, 1);

    for(int i=0; i<16; i++) {
        printf("%f ", host_result_tensor_ft32[i]);
    }
    std::cout << std::endl;

    free(host_output_tensor_ft32);
    free(host_result_tensor_ft32);

    return 0;
}