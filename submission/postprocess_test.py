import argparse
import json
from pathlib import Path
from evaluation.tools.merge import merge_ranked_predictions
from evaluation.tools.clean import trim_predictions

# ---- Augment vehicle-view entries for each sample ----
def generate_vehicle_augmented_jsonl(input_path, output_path):
    with open(input_path, "r") as f:
        original_lines = [json.loads(line) for line in f if line.strip()]

    augmented_lines = []
    for entry in original_lines:
        augmented_lines.append(entry)
        sample = entry.get("meta", {}).get("sample", "")
        label = entry.get("label", entry.get("meta", {}).get("label", ""))
        if sample:
            veh_entry = {
                "meta": {
                    "sample": sample,
                    "vqa_id": [f"{sample}_vehicle_view.mp4"],
                    "label": label
                },
                "vqa_id": [f"{sample}_vehicle_view.mp4"],
                "predictions": entry.get("predictions", []),
                "label": label
            }
            augmented_lines.append(veh_entry)

    with open(output_path, "w") as f:
        for line in augmented_lines:
            f.write(json.dumps(line) + "\n")

    print(f"Augmented with vehicle-view: {len(augmented_lines)} entries written to {output_path}")


# ---- Fill predictions into test VQA JSON ----
def load_results(jsonl_path):
    full_map = dict()
    env_map = dict()
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            label = data.get("label", data.get("meta", {}).get("label"))
            vqa_ids = data.get("vqa_id") or data.get("meta", {}).get("vqa_id") or []

            for pred in data.get("predictions", []):
                q = pred["question"].strip()
                a = pred.get("answer")
                if not a:
                    continue
                for vid in vqa_ids:
                    if label is not None:
                        full_map[(vid, label, q)] = a
                    env_map[(vid, q)] = a
    return full_map, env_map


def fill_test_vqa(vqa_json_path, results_jsonl, output_json):
    full_map, env_map = load_results(results_jsonl)

    with open(vqa_json_path) as f:
        vqa_data = json.load(f)

    filled = 0
    for entry in vqa_data:
        vids = entry.get("videos", [])
        if "event_phase" in entry:
            for phase in entry["event_phase"]:
                for label in phase.get("labels", []):
                    for conv in phase.get("conversations", []):
                        q = conv["question"].strip()
                        for vid in vids:
                            key = (vid, label, q)
                            if key in full_map:
                                conv["correct"] = full_map[key]
                                filled += 1
                                break
        elif "conversations" in entry:
            for conv in entry["conversations"]:
                q = conv["question"].strip()
                for vid in vids:
                    key = (vid, q)
                    if key in env_map:
                        conv["correct"] = env_map[key]
                        filled += 1
                        break

    with open(output_json, "w") as f:
        json.dump(vqa_data, f, indent=2)

    print(f"Filled VQA answers: {filled} → {output_json}")


# ---- Final Submission Generation ----
def generate_submission(vqa_filled_path, output_submission):
    with open(vqa_filled_path) as f:
        data = json.load(f)

    submission = []
    seen_ids = set()
    for entry in data:
        if "event_phase" in entry:
            for phase in entry["event_phase"]:
                for conv in phase.get("conversations", []):
                    qid = conv.get("id")
                    ans = conv.get("correct", "").lower()
                    if qid and qid not in seen_ids:
                        submission.append({"id": qid, "correct": ans})
                        seen_ids.add(qid)
        elif "conversations" in entry:
            for conv in entry["conversations"]:
                qid = conv.get("id")
                ans = conv.get("correct", "").lower()
                if qid and qid not in seen_ids:
                    submission.append({"id": qid, "correct": ans})
                    seen_ids.add(qid)

    with open(output_submission, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"Submission written: {output_submission} ({len(submission)} entries)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--vqa_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_ranks", type=int, default=8)
    args = parser.parse_args()

    base = Path(args.base_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = out_dir / "merged.jsonl"
    trimmed = out_dir / "trimmed.jsonl"
    veh_augmented = out_dir / "veh_augmented.jsonl"
    filled_vqa = out_dir / "vqa_filled.json"
    submission = out_dir / "submission.json"

    merge_ranked_predictions(base, args.num_ranks, merged)
    trim_predictions(merged, trimmed)
    generate_vehicle_augmented_jsonl(trimmed, veh_augmented)
    fill_test_vqa(args.vqa_json, veh_augmented, filled_vqa)
    generate_submission(filled_vqa, submission)
