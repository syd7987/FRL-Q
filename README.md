# FRL-Q

Federated reinforcement-learning experiments using FedAdagrad for vehicle path tracking.

## Completed experiments

All experiments used seed 41 and trained for 1,000 episodes.

| Experiment | Client selection | Final episode mean lateral error | Last 100 episode mean |
| --- | --- | ---: | ---: |
| 4 clients | all 4/4 | 0.3743 | 0.4363 |
| 8 clients | error-low 4/8 | 0.6502 | 0.5492 |
| 8 clients | random 4/8 | 0.3712 | 0.4392 |

For the 8-client comparison, random selection produced a lower final and late-training error than error-low selection. Error-low selection concentrated 89.5% of aggregation selections on agents 1, 4, 5, and 7, leading to weaker performance on difficult agents.

The reported values are training metrics from one seed, not held-out evaluation results.

See [EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) for the training-time analysis and raw learning curve.

## Run

Create a Python environment with the required TensorFlow stack, then run all three experiments in parallel:

```powershell
.\run_fast_parallel_experiments.ps1
```

The launcher enables mixed-precision GPU training, constrains per-worker CPU threads, staggers process startup, and saves checkpoints every five federated rounds.

## Results

Each completed experiment directory contains:

- `metrics/`: per-episode training metrics and client-selection records
- `logs/`: captured standard output and error logs
- `models/`: final trained model
- `checkpoints/`: global model weights saved during training
- `source_snapshot/`: exact source used for the run

Temporary worker files, runtime state, virtual environments, and interrupted pre-optimization runs are intentionally excluded from Git.

