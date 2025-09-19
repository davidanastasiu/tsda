import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter

def load_gt_from_val_dirs(root_dirs: List[str]) -> Dict[str, Dict]:
    gt = {}
    for root_dir in root_dirs:
        root_dir = Path(root_dir)
        for sample_dir in root_dir.rglob("*"):
            if not sample_dir.is_dir():
                continue
            sample_name = sample_dir.name
            for view in ['environment', 'overhead_view', 'vehicle_view']:
                file_path = sample_dir / view / f"{sample_name}.json"
                if not file_path.exists():
                    continue
                try:
                    data = json.load(open(file_path))
                except Exception as e:
                    print(f"Failed to read {file_path}: {e}")
                    continue
                if view == "environment":
                    qa_list = data[0]["environment"]
                    for i, qa in enumerate(qa_list):
                        qid = f"{sample_name}/environment/{i}"
                        gt[qid] = {
                            "question": qa["question"].strip(),
                            "correct": qa["correct"].lower(),
                            "view": view,
                            "sample": sample_name
                        }
                else:
                    for phase in data[0]["event_phase"]:
                        label = phase["labels"][0]
                        for i, convo in enumerate(phase["conversations"]):
                            qid = f"{sample_name}/{view}/{label}/{i}"
                            gt[qid] = {
                                "question": convo["question"].strip(),
                                "correct": convo["correct"].lower(),
                                "view": view,
                                "sample": sample_name,
                                "label": label
                            }
    return gt

def convert_predictions(pred_path: str, gt: Dict[str, Dict], output_path: str):
    q_lookup = {}
    for qid, meta in gt.items():
        key = (meta["sample"], meta.get("label", ""), meta["question"])
        q_lookup[key] = qid

    env_answers = defaultdict(list)
    converted = []

    with open(pred_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            sample = entry["meta"]["sample"]
            label = entry["meta"].get("label", "")
            for qa in entry["predictions"]:
                question = qa["question"].strip()
                answer = qa["answer"].strip().lower()
                key = (sample, label, question)
                qid = q_lookup.get(key) or q_lookup.get((sample, "", question))
                if not qid:
                    print(f"[Warning] Unmatched: ({sample}, {label}, {question})")
                    continue
                if "/environment/" in qid:
                    env_answers[(sample, qid)].append(answer)
                else:
                    converted.append({"id": qid, "correct": answer})

    for (sample, qid), answers in env_answers.items():
        majority_answer = Counter(answers).most_common(1)[0][0]
        converted.append({"id": qid, "correct": majority_answer})

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)
    print(f"🎯 Converted predictions saved to {output_path}")
