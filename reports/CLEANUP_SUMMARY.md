# Cleanup summary

One-time cleanup completed. Counts use rg --files -uu (including Git metadata and retained datasets): before **227082**; after **225610**.

Removed obsolete stage1/stage2 execution scripts, Example/32-frame configs and caches, extraction/converter/path-guard/safety-test tools, intermediate logs and reports, bytecode and test caches, historical R50 prompt, and the duplicate linked stage2 source worktree. Full data, checkpoints, official split, final val cache and both source repositories are retained.

Moved:
- Official full infos: `stage2/data_infos/V2X-Seq-SPD-CoopTrack-full` -> `data_infos/V2X-Seq-SPD-CoopTrack-full`.
- Canonical 3316-frame val cache: `stage2/artifacts/stage3/query_cache/val` -> `query_cache/val`.
- Stage2/Stage3 final reports -> `reports/`; FULL_DATA_READY report/manifest, official protocol, checkpoint/load identity and val cache manifest -> `artifacts/core/`.
- All 11 completed communication cases, including predictions and metrics -> `artifacts/core/communication_baselines/`.
- R50 alignment final results and publication evidence -> `artifacts/core/r50_alignment/`.
- Final IDS evidence, including all eleven completed runs, event review, provenance and validation -> `artifacts/core/ids_diagnosis/`; supporting reports -> its `reports/` directory. `reports/IDS_ALIGNMENT_FINAL.md` remains.
- Selected final data/diagnostic evidence -> `artifacts/core/{stage1,stage2,stage3}/`; obsolete attempts and duplicate snapshots removed.
- Existing Docker validation probes -> `docker/probes/`; installation/runtime provenance -> `artifacts/core/docker_provenance/`; references updated.
- Current inverse compatibility patch -> `patches/40xx_cusolver_compat.patch`.
- Current experiment prompt -> `configs/main_experiment/protocol.md`.

Final structure: `code/{CoopTrack,mmdetection3d}`, `configs/main_experiment`, `checkpoints`, `datasets`, `data`, `data_infos`, `query_cache/{train,val}`, `experiments/main/{checkpoints,predictions,communication,metrics}`, `scripts/main_*.py`, `scripts/run_main.py`, final `reports`, `artifacts/core`, `docker`, `patches/40xx_cusolver_compat.patch`, and `logs/main`.

Main entry: `python scripts/run_main.py --resume`. `main_train.py` and `main_eval.py` run inside the inherited Docker environment. The required mmdetection3d source/compiled operators and existing Docker rebuild files remain. Original final result JSON/prediction bytes were moved without modification; historical path strings inside evidence are preserved as provenance.