#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace std;

// 计算 dot product
float dot_product(const vector<float>& a, const vector<float>& b) {
    assert(a.size() == b.size());
    float result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// softmax
vector<float> softmax(const vector<float>& logits) {
    vector<float> result(logits.size());
    float max_logit = *max_element(logits.begin(), logits.end());
    float sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = exp(logits[i] - max_logit);
        sum += result[i];
    }
    for (float& val : result) {
        val /= sum;
    }
    return result;
}

// decoder attention 计算
vector<float> decoder_attention(
    const vector<float>& q,
    const vector<vector<float>>& k_mat,
    const vector<vector<float>>& v_mat
) {
    size_t d_k = q.size();
    size_t seq_len = k_mat.size();

    // 1. 计算 attention scores (q·k / sqrt(d_k))
    vector<float> scores(seq_len);
    for (size_t i = 0; i < seq_len; ++i) {
        scores[i] = dot_product(q, k_mat[i]) / sqrt(d_k);
    }
 
    // 2. softmax
    vector<float> attn_weights = softmax(scores);

    // 3. 加权求和 V
    vector<float> output(v_mat[0].size(), 0.0);
    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < v_mat[0].size(); ++j) {
            output[j] += attn_weights[i] * v_mat[i][j];
        }
    }

    return output;
}

// src: (len_tile,)
// value: (len_tile, size_per_head)
// => (size_per_head,)
std::tuple<float, vector<float>> onlineSoftmaxWithDot1PASS(
    std::vector<float> &src, vector<vector<float>> &value) {

    size_t len_tile = src.size();
    size_t size_per_head = value[0].size();

    vector<float> lse_vector(size_per_head);
    vector<float> O_local_vector(size_per_head);

    float sum_for_lse = 0.f;
    for (int i = 0; i < len_tile; i++) {
        sum_for_lse += std::exp(src[i]);
    }
    float lse = logf(sum_for_lse);

    for (int loop = 0; loop < size_per_head; loop++) {
        float O_local = 0.f;
        float max_value = -INFINITY;
        float pre_max_value = 0.f;
        float pre_sum = 0.f;
        float sum = 0.f;

        for (int i = 0; i < len_tile; i++) {

          max_value = std::max(max_value, src[i]);
          sum = sum * std::exp(pre_max_value - max_value) +
                std::exp(src[i] - max_value);

          O_local = O_local * pre_sum * std::exp(pre_max_value - max_value) / sum +
                std::exp(src[i] - max_value) / sum * value[i][loop];

          pre_max_value = max_value;
          pre_sum = sum;

        }
        O_local_vector[loop] = O_local;
    }

  return std::make_tuple(lse, O_local_vector);
}

std::tuple<float, std::vector<float>> onlineSoftmaxWithDot1PASS_v2(
    const std::vector<float> &src, const std::vector<std::vector<float>> &value) {

    size_t len_tile = src.size();
    size_t size_per_head = value[0].size();

    float max_value = -INFINITY;
    float pre_max_value = -INFINITY;
    float sum = 0.f;

    std::vector<float> O_prime(size_per_head, 0.f);

    for (int i = 0; i < len_tile; i++) {
        float x_i = src[i];

        max_value = std::max(max_value, x_i);
        float exp_new = std::exp(x_i - max_value);
        float exp_scale = (i == 0) ? 0.f : std::exp(pre_max_value - max_value);
        sum = sum * exp_scale + exp_new;

        for (int j = 0; j < size_per_head; j++) {
            O_prime[j] = O_prime[j] * exp_scale + exp_new * value[i][j];
        }

        pre_max_value = max_value;
    }

    // Compute logsumexp (lse)
    float lse = logf(sum) + max_value;

    // Recover final O by dividing by sum
    std::vector<float> O_local(size_per_head);
    for (int j = 0; j < size_per_head; j++) {
        O_local[j] = O_prime[j] / sum;
    }

    return std::make_tuple(lse, O_local);
}

vector<float> flash_decoding(
    const vector<float>& q,
    const vector<vector<float>>& k_mat,
    const vector<vector<float>>& v_mat
) {
    size_t d_k = q.size();
    size_t seq_len = k_mat.size();

    // 对 k v 进行分块计算
    const size_t len_tile = 2;
    const size_t loops = (seq_len + len_tile - 1) / len_tile;

    // 块1
    vector<float> qk1(len_tile);
    // (1, size_per_head) * (size_per_head, len_tile) =  (1, len_tile)
    for (size_t i = 0; i < len_tile; i++) {
        qk1[i] = dot_product(q, k_mat[i]) / sqrt(d_k);
    }

    // (1, len_tile) * (len_tile, size_per_head) = (1, size_per_head)
    // 切 v
    std::vector<std::vector<float>> v1 = {v_mat[0], v_mat[1]};
    auto [lse1, O_local1] =  onlineSoftmaxWithDot1PASS_v2(qk1, v1);

    // 块2
    vector<float> qk2(len_tile);
    // (1, size_per_head) * (size_per_head, len_tile) =  (1, len_tile)
    for (size_t i = len_tile; i < 2 * len_tile; i++) {
        qk2[i-2] = dot_product(q, k_mat[i]) / sqrt(d_k);
    }

    // (1, len_tile) * (len_tile, size_per_head) = (1, size_per_head)
    // 切 v
    std::vector<std::vector<float>> v2 = {v_mat[2], v_mat[3]};

    auto [lse2, O_local2] =  onlineSoftmaxWithDot1PASS_v2(qk2, v2);

    // reduction
    vector<float> output(d_k);

    for (int i = 0; i < d_k; i++) {
        float LSE_final = logf(std::exp(lse1) + std::exp(lse2));
        float O_final = std::exp(lse1 - LSE_final) * O_local1[i] + std::exp(lse2 - LSE_final) * O_local2[i];

        output[i] = O_final;
    }

    return output;
}

// 测试
int main() {
    // dim = 4，q_len = 1，k_len = 4
    vector<float> q = {2.0f, 2.55f, 2.2f, 4.3f};
    vector<vector<float>> k = {
        {2.2, 4.1, 2.1, 1.3},
        {4.01, 1.3, 3.4, 2.1},
        {2.1, 1.2, 3.1, 1.01},
        {1.01, 4.2, 2.1, 2.3}
    };
    vector<vector<float>> v = {
        {1.0, 0.1, 0.1, 0.2},
        {0.1, 1.0, 0.2, 0.3},
        {0.3, 0.3, 1.0, 0.4},
        {0.4, 0.5, 0.4, 1.1}
    };

    vector<float> output = decoder_attention(q, k, v);

    cout << "Output: ";
    for (float val : output) {
        cout << val << " ";
    }
    cout << endl;

    vector<float> output_2 = flash_decoding(q, k, v);

    cout << "Output2: ";
    for (float val : output_2) {
        cout << val << " ";
    }
    cout << endl;
    return 0;
}