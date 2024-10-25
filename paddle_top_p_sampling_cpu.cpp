#include <iostream>
#include <vector>
#include <algorithm>
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdint>

// Helper function to find the minimum in a non-parallel way
template<typename T>
void atomicMin(T* addr, T value) {
    *addr = std::min(*addr, value);
}

// Simulates block reduction in CUDA
template<typename T>
T blockReduceSum(const std::vector<T>& values) {
    return std::accumulate(values.begin(), values.end(), T(0));
}

// Simulates block scan in CUDA (inclusive scan)
template<typename T>
void inclusiveScan(const std::vector<T>& input, std::vector<T>& output) {
    std::partial_sum(input.begin(), input.end(), output.begin());
}

// CPU version of the sampling from probability function without VEC_SIZE slicing
template <typename T, typename IdType>
void DeviceSamplingFromProbCPU(
    uint32_t i, uint32_t d, T threshold, T u,
    std::vector<T>& prob_vec, T& aggregate, IdType& sampled_id,
    std::vector<T>& inclusive_cdf, bool deterministic) {

    std::vector<T> prob_greater_than_threshold(d, T(0));
    std::vector<bool> valid(d, false);
    
    for (uint32_t j = 0; j < d; ++j) {
        prob_greater_than_threshold[j] = (prob_vec[j] > threshold) ? prob_vec[j] : T(0);
        valid[j] = prob_vec[j] > threshold;
    }
    
    T aggregate_local = blockReduceSum(prob_greater_than_threshold);
    aggregate_local = aggregate_local;  // Simulate shared memory behavior

    if (aggregate + aggregate_local > u) {
        if (deterministic) {
            inclusiveScan(prob_greater_than_threshold, inclusive_cdf);
        } else {
            inclusiveScan(prob_greater_than_threshold, inclusive_cdf);
        }

        for (uint32_t j = 0; j < d; ++j) {
            if (inclusive_cdf[j] + aggregate > u && valid[j]) {
                if (deterministic) {
                    sampled_id = std::min(sampled_id, static_cast<int>(j)); // 将 j 转换为 int
                } else {
                    // atomicMin(&sampled_id, j);
                }
                break; // different
            }
        }
    }
    aggregate += aggregate_local;   // NO USE!!
}

// 这个函数在 probs_vec 中判断满足 pivot 的候选项并更新 aggregate 和 sampled_id。
// inclusive_cdf 存储每个元素的累计概率，用于判断是否满足 u。
// sampled_id 是当前批次的采样结果。

// q 可以理解为剩余的总概率

// CPU version of the top-p sampling kernel without VEC_SIZE slicing
template <typename T, typename IdType>
void TopPSamplingFromProbKernelCPU(std::vector<T>& probs, std::vector<T>& uniform_samples,
                                   std::vector<IdType>& output, std::vector<float>& top_p_val,
                                   uint32_t d, uint32_t max_top_p_rounds, bool deterministic) {

    uint32_t batch_size = top_p_val.size();
    
    for (uint32_t bx = 0; bx < batch_size; ++bx) {
        float top_p = top_p_val[bx];
        T q = T(1);
        T pivot = T(0);


        IdType sampled_id = d - 1;

        // 逐步将结果收敛
        for (uint32_t round = 0; round < max_top_p_rounds; ++round) {

            T u = uniform_samples[round * batch_size + bx] * q;    // q 会 change
            T aggregate = T(0);
            std::vector<T> inclusive_cdf(d, T(0));

            std::cout << "uniform_samples: " << uniform_samples[round * batch_size + bx] << std::endl;
            std::cout << "pivot: " << pivot << std::endl;
            std::cout << "u: " << u << std::endl;

            // Entire d elements are processed in a single loop
            std::vector<T> probs_vec(d, T(0));
            std::copy(probs.begin() + bx * d, probs.begin() + (bx + 1) * d, probs_vec.begin());

            DeviceSamplingFromProbCPU<T, IdType>(0, d, pivot, u, probs_vec, aggregate, sampled_id, inclusive_cdf, deterministic);

            // 更新 pivot 为 sampled_id 对应的概率值，以保证下轮仅考虑比 sampled_id 更高概率的元素，逐步收敛候选项 ！！！
            // Update pivot
            pivot = std::max(pivot, probs[bx * d + sampled_id]);

            // Compute aggregate greater than pivot
            T aggregate_gt_pivot = T(0);
            for (uint32_t j = 0; j < d; ++j) {
                if (probs[bx * d + j] > pivot) {
                    aggregate_gt_pivot += probs[bx * d + j];
                }
            }
            q = aggregate_gt_pivot;

            if (q > 0 && q < top_p) {
                break;
            }
        }
        output[bx] = sampled_id;
    }
}

// Main function for CPU top-p sampling
template <typename T, typename IdType>
void TopPSamplingFromProbCPU(std::vector<T>& probs, std::vector<T>& uniform_samples,
                             std::vector<IdType>& output, uint32_t batch_size,
                             std::vector<T>& top_p_val, uint32_t d,
                             uint32_t max_top_p_rounds, bool deterministic) {

    TopPSamplingFromProbKernelCPU<T, IdType>(probs, uniform_samples, output, top_p_val, d, max_top_p_rounds, deterministic);
}

// Example usage
int main() {
    std::vector<float> probs = {0.1f, 0.1f, 0.2f, 0.6f};
    std::vector<float> uniform_samples = {0.8f, 0.8f, 0.8f, 0.7f, 0.5f};
    std::vector<int> output(2);
    std::vector<float> top_p_val = {0.5f, 0.5f};
    uint32_t d = 4;
    uint32_t batch_size = 2;
    uint32_t max_top_p_rounds = 5;
    bool deterministic = true;

    TopPSamplingFromProbCPU<float, int>(probs, uniform_samples, output, batch_size, top_p_val, d, max_top_p_rounds, deterministic);

    for (auto id : output) {
        std::cout << "Sampled ID: " << id << std::endl;
    }

    return 0;
}
