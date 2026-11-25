# Infrastructure: where experiments run and where the data lives

This file describes the structure without addresses. The machines' addresses and the operational scripts stay outside the repository (which is public), on the control PC under `~/ml2_backup/`.

## Machines

- **Lab cluster:** 10 machines, each with one RTX A4500 (20 GB), 32 threads and about 430 GB of free disk. SSH access over the Tailscale network, no sudo. In the data they are numbered 1 to 10, in the order of `~/ml2_backup/hosts.txt` on the control PC.
- **Control PC:** coordination only (queues, health checks, inventory page). **It keeps no experiment data.**

Typical times per run: about 1.3 s per step, roughly 55 minutes for 2,500 steps and 30 to 60 minutes of evaluation; about 9 GB of GPU memory.

## On each machine (`~/ml2/`)

| Path | Contents |
|---|---|
| `repo/` | copy of the code, with processed data in `repo/data/processed/` and starting checkpoints in `repo/ckpt/` |
| `repo/experiments/tacotron2-vae/cov_<run>/` | training: checkpoints every 500 steps, `hparams.json`, `metrics.jsonl` (losses and KL per step), TensorBoard, samples |
| `out/<run>/step_<steps>/strength_1.0/` | evaluation: `results.csv`, `embeddings.npz`, `voice_bits.json`, audio |
| `out/progress.txt` | one line per event (training, evaluating, finished) |
| `node_queue.txt` | this machine's job queue, one job per line: `<run> <steps> [<steps> ...]` |
| `worker.sh` | takes the next job from the queue and runs it (`run_node.sh` → `scripts/coverage/colab_job.sh`); works with the PC off |
| `drive_up.sh` | uploads this machine's data to Google Drive every 20 minutes |

Runs from before 2026-11-24 02:30 kept only their last checkpoints; the intermediate ones of those runs are in the GCS copy. Since then, every checkpoint is kept.

## Queues

1. The main queue lives on the control PC (`~/ml2_backup/queue.txt`, then `backlog.txt`), in priority order.
2. The PC dispatcher (service `ml2-dispatch`) tops each machine's queue up to 6 jobs, one job per machine per pass, to spread the priorities. It also makes sure `worker.sh` and `drive_up.sh` are running.
3. A job that **continues** an existing run (more steps) goes straight into the queue of the machine holding the checkpoint.
4. If the main queue and the backlog run out, the PC writes `QUEUE EMPTY` to `~/ml2_backup/alerts.log`. A machine that goes down produces `DOWN` in the same file.

The PC's `ml2-sync` service checks the machines every 2 minutes (`~/ml2_backup/status.txt`) and rebuilds the inventory page (`~/ml2_backup/inventory.html`) every ~10 minutes. The page lists each experiment with status, machine, results, checkpoints and paths, plus the queues.

## Where the data is

| Place | What it holds | Note |
|---|---|---|
| Lab machines | everything (primary source) | paths above |
| Google Drive, folder `ml2_lab_cluster/<machine>/` | `experiments/`, `out/`, `logs/` of each machine | uploaded by the machines themselves (rclone, append only). The Drive login expires every 7 days (app in testing mode): renew with `rclone config reconnect gdrive:` on the PC and copy `rclone.conf` to the machines |
| The project's GCS bucket, folder `cluster/` | copy of everything up to 2026-11-24 ~02:00 (113 GB) | uploads turned off |
| This repository | results tables (`research/results/data/`) | no checkpoints or audio |

## Retrieving everything from a machine

```
rsync -a -e 'ssh -p 2222' <user>@<machine>:ml2/repo/experiments/ ./<machine>/experiments/
rsync -a -e 'ssh -p 2222' <user>@<machine>:ml2/out/ ./<machine>/out/
```

The inventory page shows these commands already filled in.
