## GT loader for VQA (GT captions will also be integrated)
import json
from pathlib import Path
from typing import Dict, List

## Only QAs from vqa

# def load_gt_from_val_dirs(root_dirs: List[str]) -> Dict[str, List[Dict]]:
#     gt = {}

#     for root_dir in root_dirs:
#         root_dir = Path(root_dir)
#         for sample_dir in root_dir.rglob("*"):
#             if not sample_dir.is_dir():
#                 continue
#             sample_name = sample_dir.name

#             for view in ['environment', 'overhead_view', 'vehicle_view']:
#                 file_path = sample_dir / view / f"{sample_name}.json"
#                 if not file_path.exists():
#                     continue

#                 try:
#                     data = json.load(open(file_path))
#                 except Exception as e:
#                     print(f"Failed to read {file_path}: {e}")
#                     continue

#                 if view == "environment":
#                     for i, qa in enumerate(data[0].get("environment", [])):
#                         gt.setdefault((sample_name, "environment"), []).append({
#                             "question": qa["question"].strip(),
#                             "correct": qa["correct"].lower()
#                         })
#                 else:
#                     for phase in data[0].get("event_phase", []):
#                         label = phase["labels"][0]
#                         for i, convo in enumerate(phase.get("conversations", [])):
#                             gt.setdefault((sample_name, label), []).append({
#                                 "question": convo["question"].strip(),
#                                 "correct": convo["correct"].lower()
#                             })
#     return gt

## QAs + options from vqa

def load_gt_from_val_dirs(root_dirs: List[str]) -> Dict[str, List[Dict]]:
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
                    for qa in data[0].get("environment", []):
                        options = {k: v for k, v in qa.items() if k.lower() in ['a', 'b', 'c', 'd']}
                        gt.setdefault((sample_name, "environment"), []).append({
                            "question": qa["question"].strip(),
                            "correct": qa["correct"].lower(),
                            "options": options
                        })

                else:
                    for phase in data[0].get("event_phase", []):
                        label = phase["labels"][0]
                        for convo in phase.get("conversations", []):
                            options = {k: v for k, v in convo.items() if k.lower() in ['a', 'b', 'c', 'd']}
                            gt.setdefault((sample_name, label), []).append({
                                "question": convo["question"].strip(),
                                "correct": convo["correct"].lower(),
                                "options": options
                            })

    return gt
