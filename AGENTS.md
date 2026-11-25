# AGENTS.md

A guide to the project for whoever works on it, human or agent. Read this before changing anything.

## What the project is

It started as the final project of ML2 (IMPA Tech): a **Tacotron 2 with a VAE** that reads a text in the style (voice, pitch, rhythm) of a reference recording. The long-term goal is **dubbing films**.

The current research asks: **how many training voices does it take to clone a new voice?** We measure how many bits of the reference voice's identity reach the generated audio and how that grows with the number of training speakers (N). So far, the bottleneck is in the decoder, and the amount of voice copied grows regularly with N (about 0.3–0.4 bit per doubling). The current state is in [`research/README.md`](research/README.md).

## Repository map

```
AGENTS.md                 this guide
README.md                 project overview (TTS results and a summary of the coverage experiment)
research/                 ALL of the research organization (start at research/README.md)
  README.md               central question, current state, how to navigate
  protocol.md             how runs are set up and evaluated (what each number means)
  infrastructure.md       where experiments run, queues, where the data lives
  results/                one .md per result (R01, R02, ...), indexed in results/README.md
    data/                 raw tables (CSV) the numbers come from
  paths/                  lines of inquiry; each has a README.md (history and status)
    <path>/in_progress/   what is being explored NOW on that path (plans, registered predictions)
src/
  models/tacotron2_vae/   model (Tacotron 2 + style VAE; optional FiLM in the decoder)
  training/training-tacotron2-vae/   training (train.py, train_utils.py)
scripts/
  coverage/               coverage experiment: train+eval job, manifests, metrics, tables
    colab_job.sh          runs one training run and its evaluation (run name -> configuration)
    coverage_manifest.py  builds the training set (LJSpeech + N speakers x K sentences)
    speaker_similarity.py "copied voice" (Resemblyzer) and embeddings.npz
    voice_bits.py         voice bits reaching the output (cross-entropy and Fano bounds)
    results_table.py      grouped table from research/results/data/cluster_runs.csv
  eval/                   synthesis and speech-recognition checks (Whisper)
  preprocess/, download_datasets/, cloud/   data and cloud execution
reports/                  longer write-ups that predate research/ (theory note, proposal, report)
  theory/coverage_theory.md   theory note with predictions registered up to section 8.12
  theory/                 theory reports (literature, derivation, information, data)
  proposal/               research proposal (PDF)
docs/                     project page and older descriptions
```

## Working rules

**Commits and GitHub**
- The only commit author is the project owner (Richard Viana). No `Co-Authored-By` or other attribution lines.
- Work on the `claude/film-dubbing-project-amuqaf` branch. Do not open a pull request unless asked.
- The repository is **public**: never write IP addresses, machine user names, e-mails, passwords, tokens or personal paths into any tracked file.

**Language**
- **Everything in the project is in English**: documents, research files, code, comments, commit messages, pages and tool output. When you touch an older Portuguese file, translate it.

**Research**
- **Prediction before result.** Before launching an experiment, write in the path's `in_progress/` file what each theory predicts and the decision rule. A prediction written afterwards does not count.
- **Each result becomes a file** in `research/results/` (next free number: see the index), with question, setup, table, conclusion, caveats and the data source. Then update the index, the path's README and, if it changed, the state in `research/README.md`.
- **Numbers come from the data.** Update `research/results/data/cluster_runs.csv` and generate the tables with `python scripts/coverage/results_table.py`. Do not copy numbers by hand without checking.
- **Negative results are results too**: record them the same way.
- State how many seeds stand behind each number. A single seed is an indication, not a conclusion.

**Cluster** (details in [`research/infrastructure.md`](research/infrastructure.md))
- Experiment data stays **on the lab machines** and is uploaded to Google Drive by the machines themselves. Do not copy data to the control PC.
- A script that a job may be using (for example `colab_job.sh` on the machines) is only replaced by writing a `.new` file and running `mv`; editing it in place breaks the bash process reading it.
- Never let the GPUs sit idle: check the queues (`node_queue.txt` on each machine) and refill them before they run out.
- Do not kill processes you did not start.
