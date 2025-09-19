import json
from argparse import ArgumentParser
from pathlib import Path
from evaluation.tools.convert import load_gt_from_val_dirs


def compute_accuracy(pred_path, gt, group_qa=False, dataset_filter=None):
    with open(pred_path) as f:
        preds = json.load(f)

    pred_map = {
        entry['id']: entry['correct'].lower()
        for entry in preds if entry.get("correct") is not None
    }

    stats = {
        "bdd": {"correct": 0, "total": 0, "missing": 0},
        "wts": {"correct": 0, "total": 0, "missing": 0}
    }

    if group_qa:
        for qid, pred in pred_map.items():
            gt_info = gt.get(qid)
            if not gt_info:
                continue
            dataset = "bdd" if gt_info["sample"].startswith("video") else "wts"
            if dataset_filter and dataset != dataset_filter:
                continue
            stats[dataset]["total"] += 1
            if pred == gt_info["correct"]:
                stats[dataset]["correct"] += 1
    else:
        for qid, info in gt.items():
            dataset = "bdd" if info["sample"].startswith("video") else "wts"
            if dataset_filter and dataset != dataset_filter:
                continue
            stats[dataset]["total"] += 1
            pred = pred_map.get(qid)
            if pred is None:
                stats[dataset]["missing"] += 1
            elif pred == info["correct"]:
                stats[dataset]["correct"] += 1

    results = {}
    for ds, s in stats.items():
        if dataset_filter and ds != dataset_filter:
            continue
        acc = 100.0 * s["correct"] / s["total"] if s["total"] else 0
        results[ds] = {
            "accuracy": acc,
            "correct": s["correct"],
            "total": s["total"],
            "missing": s["missing"]
        }

    return results


def detect_dataset_type(filename: str):
    name = filename.lower()
    if "wts" in name and "bdd" not in name:
        return "wts"
    elif "bdd" in name and "wts" not in name:
        return "bdd"
    else:
        return None  # both datasets


def evaluate_all(pred_dir: str, gt_dirs: list, multi_agent_path: str = None):
    pred_files = sorted(Path(pred_dir).glob("*.json")) if pred_dir else []
    if multi_agent_path:
        pred_files.append(Path(multi_agent_path))

    if not pred_files:
        print("No prediction files found. Provide --pred_dir or --multi_agent_path.")
        return

    gt = load_gt_from_val_dirs(gt_dirs)

    print(f"\nEvaluating {len(pred_files)} model(s)")
    print(f"Mode auto-detection: Group QA if 'group' in filename\n")

    for file in pred_files:
        is_group_qa = "group" in file.name.lower()
        dataset_filter = detect_dataset_type(file.name)
        results = compute_accuracy(file, gt, group_qa=is_group_qa, dataset_filter=dataset_filter)

        print(f"\n Model: {file.name}")
        print(f" Mode: {'Group QA' if is_group_qa else 'Full QA'}")

        total_correct = 0
        total_total = 0

        for ds, stats in results.items():
            print(f"Dataset: {ds.upper()}")
            print(f"Total Evaluated: {stats['total']}")
            print(f"Correct: {stats['correct']}")
            print(f"Incorrect: {stats['total'] - stats['correct']}")
            if not is_group_qa:
                print(f"Missing Predictions: {stats['missing']}")
            print(f"🎯 Accuracy: {stats['accuracy']:.2f}%")
            total_correct += stats["correct"]
            total_total += stats["total"]

        if dataset_filter is None or is_group_qa:
            overall_acc = 100.0 * total_correct / total_total if total_total else 0
            print(f"\nOverall Accuracy (WTS + BDD): {overall_acc:.2f}%")

        print("-" * 50)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pred_dir", type=str, help="Directory with *_converted.json files")
    parser.add_argument("--multi_agent_path", type=str, help="Optional single multi-agent results .json file")
    parser.add_argument("--gt_dirs", nargs="+", required=True, help="List of GT root dirs")
    args = parser.parse_args()

    evaluate_all(args.pred_dir, args.gt_dirs, args.multi_agent_path)
