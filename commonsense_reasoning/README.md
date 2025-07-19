# Commonsense Reasoning

This folder contains a lightweight setup to finetune and evaluate MoSLoRA on commonsense reasoning datasets.

## Usage

Simply run `run_moslora.py`:

```bash
python run_moslora.py
```

The script automatically sets up output directories and launches training followed by evaluation on several benchmarks. By default it uses the openly available model `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. You can override this by setting the environment variable `MOSLORA_BASE_MODEL` to any other checkpoint you have access to.

Training expects the dataset `ft-training_set/commonsense_170k.json` to be placed inside this folder. Results and adapter weights will be written to `trained_models/` and `results/` respectively.
If your dataset lives elsewhere, set the environment variable `MOSLORA_DATA_PATH` to its location before running the script.

## Acknowledge

This code is modified based on the [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters) project.
