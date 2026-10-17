# Three joint structure experiments

The process environment selects `REFRESH2TRACK_STRUCTURE_MODEL=A_GBA`, `B_GATE`, or `C_GBA_GATE` (default). Each process has an independent output directory under `experiments/structure_v4_joint/`. Shared native source files are never edited by the target-read optimization.

All runs retain seed 0, FP32, batch 4, original native losses, the existing communication protocol, and the registered AdamW groups and 36-epoch schedule. A trains Scheduler + original active GBA + its adapter; B trains Scheduler + five original fusion MLPs + its Gate; C trains both sets. Only the original CoopTrack checkpoint initializes the native network; the historical V4 Scheduler is used solely for step0 equivalence.

## Local

```sh
python -B scripts/structure_v4_joint/run.py --root /path/to/Refresh2Track --launch
```

The coordinator performs real-data preflight before training. Use `--resume` only with a matching completed-epoch checkpoint. Interrupted first epochs without a checkpoint restart from seed 0, not from an unrecorded partial state.

## Remote

Deploy the same scripts and the exact local train-only normalization bytes to `configs/structure_v4_joint/normalization.json` and each A/B output directory. Run `precheck.py` on the remote host and compare its complete cache, split, checkpoint, and native-source hashes with local evidence. Do not overwrite other experiments.

```sh
python -B scripts/structure_v4_joint/remote_queue.py --root /path/to/Refresh2Track --launch
```

The single-GPU queue first checks A and B independently, then trains A followed by B. It requires 16 GiB free before A and 8 GiB before B to retain full validation evidence. A storage failure leaves the queue at `BLOCKED_STORAGE`; rerun the launcher after providing space. A completed preflight is reused only when its frozen identity still matches. Do not relaunch this fresh-training queue after formal checkpoints exist; use each experiment's explicit `run.py --resume` instead.

## Performance-only optimization

`native_targets.py` converts three native target-ID loops from per-element GPU-to-CPU reads to a bulk read before each loop. It retains the original loop bodies, tensor writes, matching, loss arithmetic, gradients, and optimizer. The transform refuses unexpected native layouts and checks that restoring the read statements reconstructs the original AST exactly.

Verification commands are in `scripts/maintenance/verify_native_target_reads.py`, `verify_structure_performance.py`, and `benchmark_structure_performance.py`. The real-data equivalence test compares exact tensor bytes, gradients, optimizer states, temporal/receiver states, packets, and RNG. Finite test coverage does not establish an empirical guarantee about every future epoch or bitwise equality across different GPU models.
