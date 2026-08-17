# Compact result summaries

This folder contains compact final tables that connect the repository code to the principal
results described in the project report. Checkpoints, logs, raw predictions, training
histories and qualitative exports remain excluded.

- `architecture_loss_matrix.csv`: complete seed-42 five-architecture by ten-objective controlled panel.
- `multiseed_basic_unet.csv`: Basic U-Net summaries across seeds 7, 42, 123, 456 and 2024.
- `nnunet_objectives_100epochs_fold0.csv`: reduced seven-objective nnU-Net experiment.
- `nnunet_full_benchmark.csv`: three-fold nnU-Net ensemble and its frozen controlled U-Net comparator.

Dice and clDice values are reported on a 0–1 scale, where higher is better. The architecture
by loss matrix contains held-out test AV Dice from the fixed seed-42 panel. The multi-seed
table reports the mean and observed range across five Basic U-Net runs per objective.

The full benchmark changes several pipeline components simultaneously and is not a
single-factor architecture comparison. The reduced nnU-Net objectives are nnU-Net-specific
adaptations; the CF and soft-clDice trainers use an unweighted cross-entropy base, whereas
the corresponding controlled U-Net implementations use the study's class weighting where
documented.
