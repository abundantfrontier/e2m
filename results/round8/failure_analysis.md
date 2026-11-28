# Round 8 failure analysis

Search: {'jobs': 300, 'tried': 2988963, 'near_miss_heldout': 85, 'v3_rejects': 187602}.

Claims removed by the final checks:

- search he114_min_subarray_sum size: recheck_ok=True holdout_ok=False holdout mismatch: ref ret=7 cand ret=-9223372036854775808 status=ok {"args": [{"id": "a", "ptr": "i64", "data": [7, -9223372036854775808]}, 2], "ref_return": 7, "cand_return": -9223372036854775808, "cand_status": "ok"}
- search mbpp256_count_primes size: recheck_ok=True holdout_ok=False holdout mismatch: ref ret=172 cand ret=None status=timeout {"args": [1023], "ref_return": 172, "cand_return": null, "cand_status": "timeout"}

Speed wins whose llvm-mca estimate is not better on both CPUs:

- search mbpp100_next_palindrome: {"znver3": {"cand": 1754, "bar": 577}, "skylake": {"cand": 5302, "bar": 762}}
- search exebench_foo_23303579f55d: {"znver3": {"cand": 248, "bar": 207}, "skylake": {"cand": 388, "bar": 286}}
- search-isa mbpp6_differ_one_bit: {"znver3": {"cand": 304, "bar": 159}, "skylake": {"cand": 204, "bar": 212}}
- model he60_sum_to_n: {"znver3": {"cand": 158, "bar": 125}, "skylake": {"cand": 235, "bar": 184}}
