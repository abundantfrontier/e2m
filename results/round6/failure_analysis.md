# Round 6 failure analysis

C repair loop (c-only-7b, k=8, R=3) on 78 held-out tasks.

- Pass full tests: 64; greedy C alone passed 46 of these tasks.
- Rescued by the loop (greedy C failed, loop passed): 18 ['he114_min_subarray_sum', 'he122_add_elements', 'he139_special_factorial', 'he152_compare_scores', 'he18_how_many_times', 'he25_factorize', 'he36_fizz_buzz', 'he48_is_palindrome', 'he64_vowels_count', 'he76_is_simple_power', 'he77_iscube', 'he83_starts_one_ends', 'he9_rolling_max', 'mbpp100_next_palindrome', 'mbpp224_count_set_bits', 'mbpp256_count_primes', 'mbpp32_max_prime_factor', 'mbpp76_count_squares']
- Greedy C passed but loop's final program failed: 0 []
- Passed visible tests only (overfit to visible): 5 ['he102_choose_num', 'he104_unique_digits', 'he155_even_odd_count', 'mbpp3_is_not_prime', 'mbpp78_odd_setbit_count']
- Never passed visible tests: 9; last-attempt categories {'wrong_answer': 7, 'crash': 1, 'doesnt_assemble': 1}
- Union with the round 4 hybrid's 48 passes: 69/78
- By source: {'MBPP_n': 23, 'MBPP_pass': 18, 'HumanEval_n': 55, 'HumanEval_pass': 46}

## Untrained 7B vs fine-tuned c-only-7b (greedy C, gcc -O2)

- heldout: both pass 44; base only 19; fine-tuned only 2; neither 13. Fine-tuned categories on base-only tasks: {'wrong_answer': 14, 'timeout': 1, 'crash': 1, 'doesnt_assemble': 3}
  - base only: ['he102_choose_num', 'he114_min_subarray_sum', 'he120_k_largest', 'he122_add_elements', 'he139_special_factorial', 'he155_even_odd_count', 'he18_how_many_times', 'he25_factorize', 'he26_remove_duplicates', 'he48_is_palindrome', 'he76_is_simple_power', 'he8_sum_product', 'he9_rolling_max', 'mbpp100_next_palindrome', 'mbpp224_count_set_bits', 'mbpp256_count_primes', 'mbpp32_max_prime_factor', 'mbpp76_count_squares', 'mbpp78_odd_setbit_count']
  - fine-tuned only: ['he82_prime_length', 'mbpp244_next_square']
- val: both pass 55; base only 6; fine-tuned only 7; neither 22. Fine-tuned categories on base-only tasks: {'wrong_answer': 6}
  - base only: ['exebench_Double_0_toDouble_2_Boolean_c3764d3b3aeb', 'exebench_FMod2p_88dbae089a48', 'exebench_MAX_8ae956bfefda', 'exebench_b64e_size_762243a6d5f3', 'exebench_ehstrnlen_ad3e8426bbc9', 'exebench_findafrom_x_vi_vf_e3f4808093a3']
  - fine-tuned only: ['exebench_MUL_ASSIGN_209444b3512e', 'exebench_count_265b732dc4ac', 'exebench_digit_10_d5d63111cdfe', 'exebench_ffi_test_10_a30a09c59164', 'exebench_fifdint_34689a4f5569', 'exebench_float_check_c3bf9eeb8a05', 'exebench_foo_23303579f55d']

