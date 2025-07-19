import os
import subprocess
import sys
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    model_dir = script_dir / "trained_models"
    results_dir = script_dir / "results"

    rank = 16
    alpha = 32

    base_model = os.getenv(
        "MOSLORA_BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    data_path = Path(
        os.getenv(
            "MOSLORA_DATA_PATH",
            script_dir / "ft-training_set" / "commonsense_170k.json",
        )
    )
    output_dir = model_dir / f"moslora-r{rank}-a{alpha}-3e4"
    result_dir = results_dir / f"moslora-r{rank}-a{alpha}-3e4"

    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable,
        str(script_dir / "finetune.py"),
        "--base_model", base_model,
        "--data_path", str(data_path),
        "--output_dir", str(output_dir),
        "--batch_size", "16",
        "--micro_batch_size", "4",
        "--num_epochs", "3",
        "--learning_rate", "3e-4",
        "--cutoff_len", "256",
        "--val_set_size", "120",
        "--adapter_name", "lora",
        "--lora_r", str(rank),
        "--lora_alpha", str(alpha),
        "--use_moslora",
        "--target_modules",
        "[\"q_proj\",\"k_proj\",\"v_proj\",\"up_proj\",\"down_proj\"]",
    ]

    subprocess.run(train_cmd, check=True)

    datasets = [
        "ARC-Easy",
        "openbookqa",
        "social_i_qa",
        "ARC-Challenge",
        "winogrande",
        "piqa",
        "boolq",
        "hellaswag",
    ]

    for ds in datasets:
        eval_cmd = [
            sys.executable,
            str(script_dir / "commonsense_evaluate.py"),
            "--model", "LLaMA3",
            "--adapter", "LoRA",
            "--dataset", ds,
            "--batch_size", "1",
            "--base_model", base_model,
            "--lora_weights", str(output_dir),
            "--save_dir", str(result_dir),
        ]
        subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
