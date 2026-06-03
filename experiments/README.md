# /experiments — Experiment Logs & Results

This folder stores all outputs from the 160 main experiment runs plus ablation studies.

**Will contain:**
- `runs/` — One JSON log file per debate run (structured per the log schema: turns, scores, interventions, metadata)
- `ablation/` — Speaking-order ablation runs (reversed order)
- `analysis/` — Processed data (CSV/Parquet), statistical test results, computed metrics
- `figures/` — Generated plots and visualizations (boxplots, heatmaps, trajectory plots, etc.)
- `human_validation/` — Annotation data, inter-annotator agreement results, judge correlation scores
- Batch run scripts and experiment configuration files
