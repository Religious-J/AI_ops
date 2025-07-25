#include <iostream>
#include <math.h>
#include <vector>

std::vector<float> natvieSoftmax(std::vector<float> &src) {
  std::vector<float> dst;
  dst.resize(src.size());

  float sum = 0.f;

  for (int i = 0; i < src.size(); i++) {
    sum += std::exp(src[i]);
  }

  for (int i = 0; i < src.size(); i++) {
    dst[i] = std::exp(src[i]) / sum;
  }

  return dst;
}

std::vector<float> safeSoftmax(std::vector<float> &src) {
  std::vector<float> dst;
  dst.resize(src.size());

  float max_value = -INFINITY;
  float sum = 0.f;

  for (int i = 0; i < src.size(); i++) {
    max_value = std::max(max_value, src[i]);
  }

  for (int i = 0; i < src.size(); i++) {
    sum += std::exp(src[i] - max_value);
  }

  for (int i = 0; i < src.size(); i++) {
    dst[i] = std::exp(src[i] - max_value) / sum;
  }

  return dst;
}

std::vector<float> onlineSoftmax(std::vector<float> &src) {
  std::vector<float> dst;
  dst.resize(src.size());

  float max_value = -INFINITY;
  float pre_max_value = 0.f;
  float sum;

  for (int i = 0; i < src.size(); i++) {
    max_value = std::max(max_value, src[i]);
    sum = sum * std::exp(pre_max_value - max_value) +
          std::exp(src[i] - max_value);
    pre_max_value = max_value;
  }

  for (int i = 0; i < src.size(); i++) {
    dst[i] = std::exp(src[i] - max_value) / sum;
  }

  return dst;
}

float onlineSoftmaxWithDotSample(std::vector<float> &src,
                                 std::vector<float> &value) {

  float dst = 0.f;
  float max_value = -INFINITY;
  float pre_max_value = 0.f;
  float sum;

  for (int i = 0; i < src.size(); i++) {
    max_value = std::max(max_value, src[i]);
    sum = sum * std::exp(pre_max_value - max_value) +
          std::exp(src[i] - max_value);
    pre_max_value = max_value;
  }

  for (int i = 0; i < src.size(); i++) {
    dst += std::exp(src[i] - max_value) / sum * value[i];
  }

  return dst;
}

float onlineSoftmaxWithDot1PASS(std::vector<float> &src,
                                std::vector<float> &value) {

  float dst = 0.f;
  float max_value = -INFINITY;
  float pre_max_value = 0.f;
  float pre_sum = 0.f;
  float sum = 0.f;

  for (int i = 0; i < src.size(); i++) {
    max_value = std::max(max_value, src[i]);
    sum = sum * std::exp(pre_max_value - max_value) +
          std::exp(src[i] - max_value);
    dst = dst * (pre_sum * std::exp(pre_max_value - max_value) / sum) +
          std::exp(src[i] - max_value) / sum * value[i];
    pre_max_value = max_value;
    pre_sum = sum;
  }

  return dst;
}

std::tuple<float, float, float> onlineSoftmaxWithDot1PASS_PART(
    std::vector<float> &src, std::vector<float> &value, int begin, int end,
    float dst_, float pre_max_value_, float pre_sum_) {

  float dst = dst_;
  float max_value = pre_max_value_;
  float pre_max_value = pre_max_value_;
  float pre_sum = pre_sum_;
  float sum = pre_sum_;

  for (int i = begin; i < end; i++) {
    max_value = std::max(max_value, src[i]);
    sum = sum * std::exp(pre_max_value - max_value) +
          std::exp(src[i] - max_value);
    dst = dst * (pre_sum * std::exp(pre_max_value - max_value) / sum) +
          std::exp(src[i] - max_value) / sum * value[i];
    pre_max_value = max_value;
    pre_sum = sum;
  }

  return std::make_tuple(dst, pre_max_value, pre_sum);
}

int main() {
  std::vector<float> src = {2.0f, 2.55f, 2.2f, 4.3f, 3.25f};
  std::vector<float> dst = natvieSoftmax(src);

  for (const float element : dst) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  std::vector<float> dst1 = natvieSoftmax(src);

  for (const float element : dst1) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  std::vector<float> dst2 = onlineSoftmax(src);

  for (const float element : dst2) {
    std::cout << element << " ";
  }
  std::cout << std::endl;

  std::vector<float> val = {1.57f, 0.55f, 7.2f, 2.3f, 3.25f};
  float result = onlineSoftmaxWithDotSample(src, val);
  std::cout << result << std::endl;

  float result1 = onlineSoftmaxWithDot1PASS(src, val);
  std::cout << result1 << std::endl;

  // tiling => flash atten
  auto [result_p1, pre_max_value_p1, pre_sum_p1] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 0, 2, 0.f, 0.f, 0.f);
  auto [result_p2, pre_max_value_p2, pre_sum_p2] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 2, 5, result_p1,
                                     pre_max_value_p1, pre_sum_p1);
  std::cout << result_p2 << std::endl;

  // flash decoding D
  auto [result_d1, pre_max_value_d1, pre_sum_d1] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 0, 2, 0.f, 0.f, 0.f);
  auto [result_d2, pre_max_value_d2, pre_sum_d2] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 2, 5, 0.f, 0.f, 0.f);

  float sum_1 = 0.f;
  float sum_2 = 0.f;

  for (int i = 0; i < 2; i++) {
    sum_1 += std::exp(src[i]);
  }
  for (int i = 2; i < 5; i++) {
    sum_2 += std::exp(src[i]);
  }

  float lse_1 = logf(sum_1);
  float lse_2 = logf(sum_2);

  float result_d =
      result_d1 * std::exp(lse_1) / (std::exp(lse_1) + std::exp(lse_2)) +
      result_d2 * std::exp(lse_2) / (std::exp(lse_1) + std::exp(lse_2));

  std::cout << result_d << std::endl;

  // flash decoding F
  /**
   * For those who don't know how to do the reduction: each split i outputs O_i and LSE_i, then you can get the final output by
   *    LSE_final = log(sum(exp(LSE_i)))
   *    O_final = sum(exp(LSE_i - LSE_final) * O_i)
   * In the CUDA implementation, it do the "logsumexp trick" again -- subtract the max(LSE_i) to avoid overflow.
   */

  auto [result_f1, pre_max_value_f1, pre_sum_f1] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 0, 2, 0.f, 0.f, 0.f);
  auto [result_f2, pre_max_value_f2, pre_sum_f2] =
      onlineSoftmaxWithDot1PASS_PART(src, val, 2, 5, 0.f, 0.f, 0.f);

  // float LSE_final = logf(std::exp(lse_1) + std::exp(lse_2));
  // float O_final = std::exp(lse_1 - LSE_final) * result_d1 + std::exp(lse_2 - LSE_final) * result_d2;

  // logsumexp trick
  // log(sum(exp(LSE_i - max))) + max
  float LSE_MAX = std::max(lse_1, lse_2);
  float LSE_final = logf(std::exp(lse_1 - LSE_MAX) + std::exp(lse_2 - LSE_MAX)) + LSE_MAX;
  float O_final = std::exp(lse_1 - LSE_final) * result_d1 + std::exp(lse_2 - LSE_final) * result_d2;

  std::cout << O_final << std::endl;

  return 0;
}