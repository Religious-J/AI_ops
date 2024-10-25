/*
    TopPSamplingKernel：
    ● FillIndex
    ● setup_kernel
    ● SetCountIter
    ● DispatchKeMatrixTopPBeamTopK
      ○ KeMatrixTopPBeamTopKFt / KeMatrixTopPBeamTopK
    ● SortPairsDescending x 2
    ● DispatchTopPSampling
      ○ topp_sampling_ft / topp_sampling

    先进行 topk 计算选取 probs 前 k 个元素
    ● 细节点：
      a. TopPBeamTopK 为单 batch 总选取的 topk， 为不变的参数（20）
      b. k 的取值可以大于 TopPBeamTopK， 但实际计算过程也最多有 TopPBeamTopK 个
      c. TopKMaxLength 为单个 thread 要计算的 topk 数
    ● 如果在 topk 计算的过程中实现 sum_prob >= rand_top_p，则可以直接获取最终结果（topp 直接 return）
      ○ 通过 count_iter_begin[bid] += 1;  来传递，且该值会决定后续是否排序
    根据 topk 的结果，来确定之后的部分：
    利用 RadixSort 对每个 batch 里 vocab_size 数量的 probs 进行排序
    对排序后的 probs 进行 topp 操作
    ● 累积概率计算：逐一累加 sorted_probs 中的概率值，直到累积概率达到 top_p 设定的阈值。
    ● 如果没有选中任何元素，则选取第一个元素。
    ● 如果选中值小于阈值，则在 [0, threshold_id] 之间随机选一个。
    ● top_p 的 random 由接口传入
*/


// DOING
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

template <typename T>
void topp_sampling_ft_cpu(T* sorted_probs,
                          int64_t* sorted_id,
                          T* out_val,
                          int64_t* out_id,
                          const T* top_ps,
                          const T* threshold,
                          std::vector<std::mt19937>& random_generators,
                          const int p_num,
                          const int vocab_size,
                          const bool need_batch_random,
                          int* count_iter,
                          int* count_iter_begin) {
  for (int bid = 0; bid < p_num; ++bid) {
    int stop_flag = 0;
    float rand_p = top_ps[bid];
    const float threshold_now = threshold ? threshold[bid] : 0.f;

    if (count_iter_begin[bid] == count_iter[bid + 1]) {
      // 如果 topk 条件满足
      continue;
    }

    int threshold_id = 0;
    float running_sum = 0;
    int selected_id = -1;
    float selected_val = 0;

    for (int i = 0; i < vocab_size; ++i) {
      float prob = static_cast<float>(sorted_probs[bid * vocab_size + i]);
      running_sum += prob;

      if (prob >= threshold_now) {
        threshold_id = i;
      }

      if (running_sum >= rand_p && stop_flag == 0) {
        selected_id = i;
        selected_val = prob;
        stop_flag = 1;
      }
    }

    if (stop_flag == 0) {
      // 如果没有选中任何元素，则选取第一个元素
      out_id[bid] = sorted_id[bid * vocab_size];
      out_val[bid] = sorted_probs[bid * vocab_size];
      continue;
    }

    if (selected_val < threshold_now) {
      // 如果选中值小于阈值，则在 [0, threshold_id] 之间随机选一个
      int max_id = threshold_id;
      std::uniform_int_distribution<int> dist(0, max_id);

      if (need_batch_random) {
        int random_id = dist(random_generators[bid]);
        out_id[bid] = sorted_id[bid * vocab_size + random_id];
        out_val[bid] = sorted_probs[bid * vocab_size + random_id];
      } else {
        out_id[bid] = sorted_id[bid * vocab_size + max_id];
        out_val[bid] = sorted_probs[bid * vocab_size + max_id];
      }
    } else {
      out_id[bid] = sorted_id[bid * vocab_size + selected_id];
      out_val[bid] = sorted_probs[bid * vocab_size + selected_id];
    }
  }
}
// DOING

