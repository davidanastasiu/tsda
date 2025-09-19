import argparse
import json
from pathlib import Path
from collections import defaultdict

def load_gt_from_val_dirs(root_dirs):
    gt = {}
    q_lookup = defaultdict(list)

    for root_dir in root_dirs:
        root_dir = Path(root_dir)
        for sample_dir in root_dir.rglob("*"):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name

            for view in ["environment", "overhead_view", "vehicle_view"]:
                file_path = sample_dir / view / f"{sample_name}.json"
                if not file_path.exists():
                    continue

                try:
                    data = json.load(open(file_path))
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue

                if view == "environment":
                    for i, qa in enumerate(data[0]["environment"]):
                        qid = f"{sample_name}/environment/{i}"
                        gt[qid] = {
                            "question": qa["question"].strip(),
                            "answer": qa[qa["correct"]].strip(),
                            "sample": sample_name,
                            "label": "environment",
                            "view": "environment",
                        }
                        q_lookup[(sample_name, "environment")].append(qid)
                else:
                    for phase in data[0]["event_phase"]:
                        label = phase["labels"][0]
                        for i, convo in enumerate(phase["conversations"]):
                            qid = f"{sample_name}/{view}/{label}/{i}"
                            gt[qid] = {
                                "question": convo["question"].strip(),
                                "answer": convo["answers"][convo["correct"]].strip() if "answers" in convo else convo[convo["correct"]].strip(),
                                "sample": sample_name,
                                "label": label,
                                "view": view,
                            }
                            q_lookup[(sample_name, label)].append(qid)
    return gt, q_lookup

def reconstruct_jsonl(best_qa_path, gt_dirs, output_path):
    gt, q_lookup = load_gt_from_val_dirs(gt_dirs)

    with open(best_qa_path) as f:
        best_preds = json.load(f)

    pred_map = defaultdict(list)
    for entry in best_preds:
        qid = entry["id"]
        answer = entry["correct"].lower()

        if qid not in gt:
            print(f"[Warning] QID {qid} not in GT")
            continue

        sample = gt[qid]["sample"]
        label = gt[qid]["label"]
        question = gt[qid]["question"]
        full_answer = gt[qid]["answer"]

        pred_map[(sample, label)].append({
            "question": question,
            "answer": full_answer,
            "qid": qid
        })

    final_entries = []
    for (sample, label), qas in pred_map.items():
        # Skip environment-only QA lines
        if label == "environment":
            continue

        # Append environment QAs
        env_qids = q_lookup.get((sample, "environment"), [])
        for qid in env_qids:
            if qid in gt:
                qas.append({
                    "question": gt[qid]["question"],
                    "answer": gt[qid]["answer"],
                    "qid": qid
                })

        view = "vehicle" if sample.startswith("video") else "overhead"
        entry = {
            "meta": {
                "sample": sample,
                "label": label,
                "view": view
            },
            "predictions": [
                {"question": q["question"], "answer": q["answer"], "qid": q["qid"]}
                for q in qas
            ]
        }
        final_entries.append(entry)

    with open(output_path, "w") as f:
        for entry in final_entries:
            json.dump(entry, f)
            f.write("\n")

    print(f"JSONL reconstructed to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best", required=True, help="Path to best_qa_answers.json")
    parser.add_argument("--gt_dirs", nargs="+", required=True, help="List of GT dirs")
    parser.add_argument("--output", required=True, help="Output path (.jsonl)")

    args = parser.parse_args()
    reconstruct_jsonl(args.best, args.gt_dirs, args.output)
