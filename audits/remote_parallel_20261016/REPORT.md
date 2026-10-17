# Remote A/B concurrency and GPU diagnosis

## Applied change

On 2026-10-16, following the user's explicit concurrency request, the external maintenance launcher was changed from sequential A-then-B to concurrent A/B. The previous serial arrangement was not the requested execution policy.

Only supervisor PID 7141 was terminated. A coordinator PID 7352 and training worker PID 7471 continued without restart or loss of progress. Parallel supervisor PID 8293 adopted A and launched independent B coordinator PID 8294 / worker PID 8399. Both independent preflights had passed; the original coordinator rechecks the frozen identity before starting training.

The 33 protected files, including frozen training sources, base weights, normalization, and preflight records, have unchanged SHA256 hashes. FP32, batch size 4, seed 0, optimizer/LR, loss, matching, communication protocol, validation, and per-process RNG handling were not modified. The existing implementation does not enforce deterministic CUDA kernels; this operational change is not a new proof of bitwise equivalence for complete concurrent versus serial runs.

## Passive throughput measurements

| Measurement | A only, 60.031 s | A and B, 120.045 s |
| --- | ---: | ---: |
| A batches/s | 0.966167 | 0.783040 |
| B batches/s | not running | 0.733059 |
| Combined batches/s | 0.966167 | 1.516098 |
| GPU mean utilization | 16.08% | 44.86% |
| GPU maximum memory, MiB | 2580 | 5155 |

Observed aggregate batch production increased about 56.9%. A individually slowed while sharing the GPU, but both experiments now advance concurrently. These windows cover different batches and models; the percentage is an operational observation, not a matched-input algorithmic speedup claim. The previous target-ID read optimization remains deployed; its separate local controlled comparisons reported approximately 44.5% higher throughput, not a guarantee of higher GPU utilization on every machine/window.

Parallel observation covered A epoch 1 steps 895–989 and B epoch 1 steps 34–122. Both losses remained finite. Peak cgroup memory was 19.17 GiB against a 30 GiB limit; GPU capacity is 12288 MiB. No cgroup OOM or OOM-kill events occurred. The container has a 10-CPU quota, not unrestricted access to all 80 host logical CPUs. The preceding single-A window used approximately 1.94 CPU cores and had no CPU-quota throttling.

Final verification at Beijing time 21:36:59 observed A epoch 1 step 1083 and B epoch 1 step 214, both running. These are snapshots, not live counters.

## Located input-side bottleneck

After the passive measurement finished, an isolated input-only diagnostic ran 8 warmup plus 24 measured real batches. It used the existing pipeline and a fresh Scheduler without native forward, backward, optimizer updates, or checkpoint writes. It did not alter the formal A/B processes.

- Input/packet preparation averaged 0.4888 s/batch, 11.730 s total.
- `dataset.prepare_train_data` consumed 9.498 s across 96 frames: approximately 81% of input preparation time, **not** 81% of total training time.
- Image normalization: 3.290 s; native image loading: 2.550 s, including OpenCV image read/decode 2.038 s; image color augmentation: 2.220 s.
- The profile supports substantial CPU-side image processing before GPU work. The training path also retains synchronous protocol/validation checks. GPU starvation is not primarily a lack of available VRAM.
- This diagnostic ran while formal A/B remained active and includes profiler overhead. Its timings must not be directly subtracted from the separate end-to-end window to claim an exact whole-training percentage.

## Further performance-only candidates

The next candidate is bounded, ordered CPU image decode/prefetch while retaining the original augmentation and per-scene RNG order. Do not cache randomized augmented images, change the data order, or enable ordinary multi-worker loading without equivalence checks. Before deployment, compare exact batch tensors, RNG states, packets, matching, losses, gradients, optimizer state, and updated weights on identical real batches. This candidate has **not** been deployed in this change.

Do not enable AMP/TF32, change batch size, skip correctness checks, or select new cuDNN algorithms merely to raise utilization. GPU utilization measures the fraction of sampled time with kernels executing, not achieved arithmetic throughput; see [NVIDIA's definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html).

## Evidence and operations

`A_only_before.json`, `parallel120.json`, `input_profile.json`, `verification.json`, and `launcher_tests.json` contain the measurements and checks. Mocked launcher tests cover both fresh concurrent startup and adopting A without restarting it, supervisor-only termination, independent environments/locks, and owned-group low-disk pause/resume.

The parallel launcher retains the user's relaxed admission policy. Below 2 GiB free it pauses both isolated active training groups; at 3 GiB it resumes them. Full-run disk capacity is still not guaranteed. Do not launch another queue over these active coordinators.
