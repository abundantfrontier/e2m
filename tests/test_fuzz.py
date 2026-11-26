from e2m.harness.fuzz import run_fuzz
from e2m.harness.emu import emulate


def test_fuzz_smoke_never_crashes():
    report = run_fuzz(n=256, workers=2, seed=0, batch_size=32, batch_timeout_s=30)
    assert report.harness_crashes == 0, report.crash_samples
    assert report.hangs == 0, report.crash_samples
    assert report.n >= 256


def test_fuzz_10k_never_crashes():
    report = run_fuzz(n=10_000, workers=None, seed=1, batch_size=50, batch_timeout_s=60)
    assert report.n >= 10_000
    assert report.harness_crashes == 0, report.crash_samples[:3]
    assert report.hangs == 0
    assert report.runs_per_sec > 10
    # Store throughput on the report object; the test just asserts survival.
    print(f"\nfuzz throughput: {report.runs_per_sec:.1f} runs/s over {report.elapsed_s:.2f}s")
    print(f"status histogram: {report.by_status}")


def test_single_worker_rss_does_not_explode():
    # A few hundred runs in-process: RSS growth should stay modest.
    import os

    def rss():
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        return 0

    before = rss()
    for i in range(200):
        emulate(bytes([i & 0xFF, 0xC3]), signature="void f(void)", args=[], insn_limit=200, timeout_s=0.02)
    after = rss()
    # Allow 64 MiB of slop for allocator fragmentation / unicorn JIT cache.
    assert after - before < 64 * 1024, f"RSS grew {after - before} kB"
