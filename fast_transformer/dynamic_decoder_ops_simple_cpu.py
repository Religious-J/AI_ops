import numpy as np

class RepetitionPenaltyType:
    ADDITIVE = "additive"               # the presence penalty
    MULTIPLICATIVE = "multiplicative"   # the repetition penalty
    NONE = "none"                       # No repetition penalty.

def cpu_invoke_ban_bad_words(logits, output_ids, bad_words, bad_words_len, share_words, batch_size, vocab_size_padded, step):
    for batch_idx in range(batch_size):
        if share_words:
            base_bad_words = bad_words
        else:
            base_bad_words = bad_words[batch_idx * 2 * bad_words_len: (batch_idx + 1) * 2 * bad_words_len]
        base_bad_words_offsets = base_bad_words[bad_words_len:]

        for bad_word_id in range(bad_words_len):
            if base_bad_words_offsets[bad_word_id] < 0:
                break

            item_end = base_bad_words_offsets[bad_word_id]
            item_start = base_bad_words_offsets[bad_word_id - 1] if bad_word_id > 0 else 0
            item_size = item_end - item_start

            # The single-token case unconditionally bans the token
            should_ban = item_size == 1

            if item_size > 1 and step > item_size - 1:
                should_ban = True
                for token_idx in range(item_size - 2, -1, -1):
                    previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size + batch_idx]
                    if previous_token != base_bad_words[item_start + token_idx]:
                        should_ban = False
                        break

            if should_ban:
                banned_token = base_bad_words[item_end - 1]
                if 0 < banned_token < vocab_size_padded:
                    logits[batch_idx * vocab_size_padded + banned_token] = -np.inf
                    

def cpu_invoke_temperature_penalty(logits, bias, temperatures, temperatures_size, batch_size, vocab_size, vocab_size_padded):
    for batch_idx in range(batch_size):
        temperature = temperatures[batch_idx] if temperatures_size > 1 else temperatures[0]
        inv_temperature = 1.0 / (temperature + 1e-6)

        for vocab_idx in range(vocab_size):
            index = batch_idx * vocab_size_padded + vocab_idx
            logit = logits[index]
            if bias is not None:
                logit += bias[vocab_idx]
            logits[index] = logit * inv_temperature
            


def cpu_invoke_repetition_penalty(logits, output_ids, input_lengths, max_input_length, batch_size, vocab_size, vocab_size_padded, step, repetition_penalty_type, repetition_penalty):
    for batch_idx in range(batch_size):
        input_length = input_lengths[batch_idx] if input_lengths is not None else max_input_length
        repet_penalty = float(repetition_penalty)

        offset = batch_idx * vocab_size_padded
        for i in range(step - 1, -1, -1):
            # ignore input paded
            if input_length <= i < max_input_length:
                continue
            
            idx = batch_idx + i * batch_size
            token_id = output_ids[idx]

            logit = logits[offset + token_id]
            if repetition_penalty_type == RepetitionPenaltyType.ADDITIVE:
                logits[offset + token_id] = logit - repet_penalty
            elif repetition_penalty_type == RepetitionPenaltyType.MULTIPLICATIVE:
                logits[offset + token_id] = logit * repet_penalty if logit < 0.0 else logit / repet_penalty
            else:
                raise ValueError("Invalid repetition penalty type.")

def cpu_invoke_min_length_penalty(logits, min_length, end_ids, sequence_lengths, max_input_length, batch_size, vocab_size_padded):
    for batch_idx in range(batch_size):
        # We need +1 because sequence_lengths = max_input_length + num_gen_tokens - 1
        if sequence_lengths[batch_idx] + 1 - max_input_length < min_length:
            mask_val = -np.inf
            logits[batch_idx * vocab_size_padded + end_ids[batch_idx]] = mask_val

def cpu_invoke_add_bias_end_mask(logits, bias, end_ids, finished, batch_size, vocab_size, vocab_size_padded):
    for batch_idx in range(batch_size):
        finish = finished[batch_idx] if finished is not None else False
        offset = batch_idx * vocab_size_padded

        for vocab_idx in range(vocab_size_padded):
            if vocab_idx < vocab_size:
                if finish:
                    logits[offset + vocab_idx] = np.finfo(np.float32).max if vocab_idx == end_ids[batch_idx] else -np.inf
                else:
                    bias_val = bias[batch_idx] if bias is not None else 0.0
                    logits[offset + vocab_idx] += bias_val
            else:
                # padded part
                logits[offset + vocab_idx] = -np.inf

def cpu_invoke_add_bias_softmax(logits, bias, end_ids, finished, batch_size, vocab_size, vocab_size_padded):
    for batch_idx in range(batch_size):
        max_val = -np.inf
        finish = finished[batch_idx] if finished is not None else False
        offset = batch_idx * vocab_size_padded

        for vocab_idx in range(vocab_size_padded):
            if vocab_idx < vocab_size:
                if finish:
                    logits[offset + vocab_idx] = np.finfo(np.float32).max if vocab_idx == end_ids[batch_idx] else -np.inf
                else:
                    bias_val = bias[batch_idx] if bias is not None else 0.0
                    logits[offset + vocab_idx] += bias_val
            else:
                # padded part
                logits[offset + vocab_idx] = -np.inf

            logit = logits[offset + vocab_idx]
            if logit > max_val:
                max_val = logit

        sum_exp = 0.0
        for vocab_idx in range(vocab_size):
            logits[offset + vocab_idx] = np.exp(logits[offset + vocab_idx] - max_val)
            sum_exp += logits[offset + vocab_idx]

        for vocab_idx in range(vocab_size):
            logits[offset + vocab_idx] /= (sum_exp + 1e-6)

def rand_float():
    return np.random.rand()

def bubble_sort_topk(vals, indices, n, k):
    for i in range(k):
        for j in range(i + 1, n):
            if vals[j] > vals[i]:
                # Swap values
                vals[i], vals[j] = vals[j], vals[i]
                # Swap corresponding indices
                indices[i], indices[j] = indices[j], indices[i]

def cpu_invoke_batch_topk_sampling(
    log_probs,
    ids,
    sequence_lengths,
    end_ids,
    finished,
    max_top_k,
    top_ks,
    top_p,
    top_ps,
    batch_size,
    vocab_size,
    vocab_size_padded,
    cum_log_probs,
    output_log_probs,
    skip_decode
):
    
    for batch_id in range(batch_size):
        if skip_decode is not None and skip_decode[batch_id]:
            continue
        
        k = top_ks[batch_id] if top_ks is not None else max_top_k
        prob_threshold = top_ps[batch_id] if top_ps is not None else top_p
        
        if finished is not None and finished[batch_id]:
            ids[batch_id] = end_ids[batch_id]
            continue
        
        # Step 1: Perform Top-k selection directly on log_probs
        topk_vals = log_probs[batch_id * vocab_size_padded: (batch_id + 1) * vocab_size_padded].copy()
        topk_indices = np.arange(vocab_size_padded)

        # Sort the first k elements
        bubble_sort_topk(topk_vals, topk_indices, vocab_size_padded, k)

        # Step 2: Find max value for softmax pre-processing
        s_max = -np.inf
        if cum_log_probs is None and output_log_probs is None:
            s_max = max(topk_vals[:k])

        # Step 3: Softmax and normalize
        s_sum = 0.0  # Sum of top-k probabilities
        for i in range(k):
            if cum_log_probs is None and output_log_probs is None:
                topk_vals[i] = np.exp(topk_vals[i] - s_max)  # Numerically stable softmax
            s_sum += topk_vals[i]

        # Step 4: Generate random number and perform Top-k sampling
        rand_num = rand_float() * prob_threshold * s_sum
        # rand_num = 0.8 * prob_threshold * s_sum
        
        for i in range(k):
            rand_num -= topk_vals[i]
            if rand_num <= 0.0 or i == k - 1:
                ids[batch_id] = topk_indices[i]
                
                # Compute cumulative log probabilities and output log probabilities
                if cum_log_probs is not None or output_log_probs is not None:
                    log_prob = np.log(topk_vals[i])
                    if cum_log_probs is not None:
                        cum_log_probs[batch_id] += log_prob
                    if output_log_probs is not None:
                        output_log_probs[batch_id] = log_prob - np.log(s_sum)
                break

        # Step 5: Update sequence lengths and finished status
        if sequence_lengths is not None and finished is not None:
            sequence_lengths[batch_id] = sequence_lengths[batch_id] + (0 if finished[batch_id] else 1)
            finished[batch_id] = (ids[batch_id] == end_ids[batch_id])
            
def cpu_invoke_stop_words_criterion(output_ids, stop_words, finished, stop_words_len, batch_size, step):
    for batch_idx in range(batch_size):
        base_stop_words = stop_words[batch_idx * 2 * stop_words_len : (batch_idx + 1) * 2 * stop_words_len]
        base_offsets = base_stop_words[stop_words_len:]

        for id in range(stop_words_len):
            if base_offsets[id] < 0:
                continue

            item_end = base_offsets[id]
            item_start = base_offsets[id - 1] if id > 0 else 0
            item_size = item_end - item_start

            should_stop = False
            if step + 1 >= item_size:
                should_stop = True
                for token_idx in range(item_size - 1, -1, -1):
                    previous_token = output_ids[(step - (item_size - 1) + token_idx) * batch_size + batch_idx]
                    if previous_token != base_stop_words[item_start + token_idx]:
                        should_stop = False
                        break
            
            if should_stop:
                finished[batch_idx] = True


def cpu_invoke_length_criterion(finished, should_stop, finished_sum, sequence_limit_length, batch_size, step):
    finished_count = 0
    for batch_index in range(batch_size):
        finished[batch_index] |= step >= sequence_limit_length[batch_index]
        finished_count += 1 if finished[batch_index] else 0

    finished_sum[0] = finished_count
    should_stop[0] = finished_count == batch_size
    
def print_logits(logits, name):
    print(f"{name:<25}", end="")
    for logit in logits:
        print(f"{logit:<20}", end=" ")
    print()

def print_array(input_array, name):
    print(f"{name:<15}", end="")
    for item in input_array:
        print(f"{item:<5}", end=" ")
    print()
    
def cpu_custom_transformer_dynamic_decoder(
    logits,                      # [batch_size * vocab_size_padded]
    output_ids,                  # [batch_size * sequence_limit_length]
    step,                        # [1]
    batch_size,                  # [1]
    vocab_size,                  # [1]
    vocab_size_padded,           # [1]
    end_ids,                     # [batch_size]
    share_words,                 # [1]
    bad_words_list,              # [bad_words_len * 2]
    bad_words_len,               # [1]
    embedding_bias,              # [vocab_size]
    temperature,                 # [1] or [batch_size]
    temperature_size,            # [1]
    repetition_penalty,          # [1]
    penalty_type,                # [1]
    sequence_lengths,            # [batch_size]
    sequence_limit_length,       # [batch_size]
    max_input_length,            # [1]
    input_length,                # [batch_size]
    min_length,                  # [1]
    max_top_k,                   # [1]
    top_ks,                      # [batch_size]
    top_p,                       # [1]
    top_ps,                      # [batch_size]
    cum_log_probs,               # [batch_size]
    output_log_probs,            # [batch_size * (sequence_limit_length - max_input_length)]
    stop_words_len,              # [1]
    stop_words_list,             # [stop_words_len * 2]
    finished,                    # [batch_size]
    finished_sum,                # [1]
    should_stop,                 # [1]
    skip_decode                  # [batch_size]
):
    print_logits(logits, "INPUT")

    if bad_words_list is not None:
        cpu_invoke_ban_bad_words(
            logits,
            output_ids,
            bad_words_list,
            bad_words_len,
            share_words,
            batch_size,
            vocab_size_padded,
            step
        )

    print_logits(logits, "AFTER BanWords")

    if embedding_bias is not None or temperature is not None:
        cpu_invoke_temperature_penalty(
            logits,
            embedding_bias,
            temperature,
            temperature_size,
            batch_size,
            vocab_size,
            vocab_size_padded
        )

    print_logits(logits, "AFTER Temperature")

    if step > 0 and penalty_type != RepetitionPenaltyType.NONE:
        cpu_invoke_repetition_penalty(
            logits,
            output_ids,
            input_length,
            max_input_length,
            batch_size,
            vocab_size,
            vocab_size_padded,
            step,
            penalty_type,
            repetition_penalty
        )

    print_logits(logits, "AFTER Repetition")

    if step - max_input_length < min_length:
        cpu_invoke_min_length_penalty(
            logits,
            min_length,
            end_ids,
            sequence_lengths,
            max_input_length,
            batch_size,
            vocab_size_padded
        )

    print_logits(logits, "AFTER MinLength")

    cpu_invoke_add_bias_end_mask(
        logits,
        None,
        end_ids,
        finished,
        batch_size,
        vocab_size,
        vocab_size_padded
    )

    print_logits(logits, "AFTER AddBiasEndMask")

    if cum_log_probs is not None or output_log_probs is not None:
        cpu_invoke_add_bias_softmax(
            logits,
            None,
            end_ids,
            finished,
            batch_size,
            vocab_size,
            vocab_size_padded
        )

        print_logits(logits, "AFTER AddBiasSoftMax")

    cpu_invoke_batch_topk_sampling(
        logits,
        output_ids[step * batch_size:],
        sequence_lengths,
        end_ids,
        finished,
        max_top_k,
        top_ks,
        top_p,
        top_ps,
        batch_size,
        vocab_size,
        vocab_size_padded,
        cum_log_probs,
        output_log_probs[(step - max_input_length) * batch_size:],
        skip_decode
    )

    print_logits(logits, "AFTER TopKSampling")
    print_array(output_ids, "output_ids")
    print_array(finished, "finished_1")

    if stop_words_list is not None:
        cpu_invoke_stop_words_criterion(
            output_ids,
            stop_words_list,
            finished,
            stop_words_len,
            batch_size,
            step
        )
    
    print_array(finished, "finished_2")
    print_array(should_stop, "should_stop1")

    if sequence_lengths is not None:
        cpu_invoke_length_criterion(
            finished,
            should_stop,
            finished_sum,
            sequence_limit_length,
            batch_size,
            step
        )
    
    print_array(should_stop, "should_stop2")
    
def main():
    batch_size = 1
    vocab_size = 5
    vocab_size_padded = 5
    max_input_length = 4
    end_ids = np.array([4])
    share_words = True
    bad_words_list = np.array([1, 2, 2, -1])
    bad_words_len = 1
    embedding_bias = np.array([0.1] * vocab_size)
    temperature = np.array([0.5])
    temperature_size = 1
    repetition_penalty = 0.1
    penalty_type = RepetitionPenaltyType.ADDITIVE
    min_length = 3
    max_top_k = 3
    top_ks = np.array([2])
    top_p = 0.9
    top_ps = np.array([0.8])
    stop_words_list = np.array([0, 1])
    stop_words_len = 1

    logits = np.array([[0.2, 0.5, 0.1, 0.15, 0.05]], dtype=np.float32)
    step = max_input_length
    sequence_lengths = np.array([max_input_length])
    sequence_limit_length = np.array([10])
    output_ids = np.zeros(sequence_limit_length * batch_size, dtype=int)
    output_ids[:4] = [2, 1, 3, 1] 
    input_length = np.array([max_input_length])
    finished = np.array([False])
    finished_sum = np.array([0])
    should_stop = np.array([False])
    skip_decode = np.array([False])
    cum_log_probs = np.array([0.0], dtype=np.float32)
    output_log_probs = np.zeros(10, dtype=np.float32)

    cpu_custom_transformer_dynamic_decoder(
        logits.flatten(),
        output_ids,
        step,
        batch_size,
        vocab_size,
        vocab_size_padded,
        end_ids,
        share_words,
        bad_words_list,
        bad_words_len,
        embedding_bias,
        temperature,
        temperature_size,
        repetition_penalty,
        penalty_type,
        sequence_lengths,
        sequence_limit_length,
        max_input_length,
        input_length,
        min_length,
        max_top_k,
        top_ks,
        top_p,
        top_ps,
        cum_log_probs,
        output_log_probs,
        stop_words_len,
        stop_words_list,
        finished,
        finished_sum,
        should_stop,
        skip_decode
    )

if __name__ == "__main__":
    main()