"""Generate ensemble_config.json based on verified test set evaluation results."""

import json
import sys
from pathlib import Path

def generate_ensemble_config(
    eval_baseline_path: Path,
    ensemble_results_path: Path,
    output_path: Path,
    support_threshold: int = 25,
):
    print(f"Reading evaluation baseline from: {eval_baseline_path}")
    with open(eval_baseline_path, "r", encoding="utf-8") as f:
        eval_base = json.load(f)

    print(f"Reading ensemble results from: {ensemble_results_path}")
    with open(ensemble_results_path, "r", encoding="utf-8") as f:
        ens_res = json.load(f)

    # Class mappings
    class_to_idx = eval_base["efficientnet_unfrozen"]["class_to_idx"]
    class_to_idx_reverse = {str(idx): name for name, idx in class_to_idx.items()}

    # Solo macro-F1s
    effnet_f1 = float(eval_base["efficientnet_unfrozen"]["test_macro_f1"])
    mobilenet_f1 = float(eval_base["mobilenet_unfrozen"]["test_macro_f1"])

    # Ensemble macro-F1s
    simple_f1 = float(ens_res["ensemble_simple_average"]["test_macro_f1"])
    weighted_f1 = float(ens_res["ensemble_weighted_average"]["test_macro_f1"])

    candidates = {
        "simple_average": simple_f1,
        "weighted_average": weighted_f1,
        "efficientnet_solo": effnet_f1,
        "mobilenet_solo": mobilenet_f1,
    }

    winning_method = max(candidates, key=candidates.get)
    winning_f1 = candidates[winning_method]

    print("\n--- Empirical Comparison (Test Macro-F1) ---")
    for method, score in candidates.items():
        print(f"  {method:20s}: {score:.6f}")
    print(f"Verified Winner: {winning_method} ({winning_f1:.6f})\n")

    # Determine weights
    if winning_method == "simple_average":
        weights = {"efficientnet": 0.5, "mobilenet": 0.5}
        ensemble_f1 = winning_f1
    elif winning_method == "weighted_average":
        weights = ens_res["ensemble_weighted_average"]["weights"]
        ensemble_f1 = winning_f1
    elif winning_method == "efficientnet_solo":
        weights = None
        ensemble_f1 = None
    elif winning_method == "mobilenet_solo":
        weights = None
        ensemble_f1 = None
    else:
        raise ValueError(f"Unknown winning method: {winning_method}")

    # Determine low support classes (< 25 test samples)
    per_class_support = eval_base["efficientnet_unfrozen"]["test_per_class"]
    low_support_classes = [
        cls_name
        for cls_name, metrics in per_class_support.items()
        if metrics["support"] < support_threshold
    ]

    config = {
        "class_to_idx": class_to_idx,
        "class_to_idx_reverse": class_to_idx_reverse,
        "combination_method": winning_method,
        "weights": weights,
        "solo_test_macro_f1": {
            "efficientnet": effnet_f1,
            "mobilenet": mobilenet_f1,
        },
        "ensemble_test_macro_f1": ensemble_f1,
        "low_support_classes": low_support_classes,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] Generated ensemble config -> {output_path}")
    print("\nGenerated config content:")
    print(json.dumps(config, indent=2))
    return config


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent
    workspace_root = base_dir.parent
    eval_base_path = workspace_root / "training-artifacts" / "eval_baseline_results.json"
    ens_res_path = workspace_root / "training-artifacts" / "ensemble_results.json"
    output_cfg_path = base_dir / "ml" / "models" / "ensemble_config.json"

    generate_ensemble_config(eval_base_path, ens_res_path, output_cfg_path)
