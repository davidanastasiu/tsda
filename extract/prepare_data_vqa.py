# Considers all edge cases like filtering, appending extra QAs and prompts
import argparse
import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from PIL import Image
import re
import random
from collections import Counter

def merge_json_files(wts_path, bdd_path):
    with open(wts_path, 'r') as f:
        wts_data = json.load(f)
    with open(bdd_path, 'r') as f:
        bdd_data = json.load(f)
    merged = []
    for idx, entry in enumerate(wts_data + bdd_data):
        entry["id"] = idx
        merged.append(entry)
    return merged


def load_valid_views(csv_path):
    df = pd.read_csv(csv_path)
    valid_views = {}
    for _, row in df.iterrows():
        # scene key must match entry["sample"] exactly
        scene = str(row["Scene"]).strip()
        cameras = []
        for col in ["Viewpoint1", "Viewpoint2", "Viewpoint3", "Viewpoint4"]:
            v = row.get(col)
            if pd.notna(v):
                stem = Path(str(v)).stem.strip()  
                if stem:
                    cameras.append(stem)
        valid_views[scene] = cameras
    return valid_views

def filter_wts_overhead(entries, valid_views, kept_log, removed_log, cam_map_path):
    """
    Substring-based filtering (matches your JSONL script):
      - For WTS + overhead entries only:
        * Keep images whose path contains ANY of the allowed camera stems for that sample.
        * If nothing matches, DROP the entry (same as your JSONL script).
      - For all other entries (BDD or non-overhead), pass through unchanged.
    """
    filtered = []
    sample_to_camera = {}  # store matched cameras per sample (for reference/logging)

    with open(kept_log, 'w') as klog, open(removed_log, 'w') as rlog:
        for entry in tqdm(entries, desc="Filtering overhead views"):
            sample = entry.get("sample", "")
            view   = entry.get("view", "")
            images = entry.get("image", [])

           
            if not sample.startswith("video") and view == "overhead":
                allowed_cameras = set(valid_views.get(sample, []))  # stems
                if not allowed_cameras:
                    
                    entry["num_images"] = len(images)
                    filtered.append(entry)
                    continue

                
                filtered_images = [
                    path for path in images
                    if any(cam in path for cam in allowed_cameras)
                ]

                if filtered_images:
                    matched_cams = sorted({cam for cam in allowed_cameras if any(cam in p for p in filtered_images)})
                    sample_to_camera[sample] = matched_cams

                    entry["image"] = filtered_images
                    entry["num_images"] = len(filtered_images)
                    klog.write(f"{sample},{'|'.join(matched_cams)},{len(filtered_images)} images kept\n")
                    filtered.append(entry)
                else:
                    
                    rlog.write(f"{sample}: no allowed camera matched; entry dropped\n")
            else:
                
                entry["num_images"] = len(images)
                filtered.append(entry)

    with open(cam_map_path, 'w') as f:
        json.dump(sample_to_camera, f, indent=2)

    return filtered


def merge_overhead_and_vehicle(entries):
    """
    Merge WTS overhead + vehicle by (sample, label):
      - IMAGES: keep from ALL views.
      - VQA: take from OVERHEAD only; if overhead has none, take from first non-overhead.
      - vqa_id: preserve from overhead if present.
      - BDD: pass through unchanged.
    """
    merged_map = defaultdict(lambda: {
        "image": [],
        "view": "overhead",
        "vqa": [],
        "vqa_id": None,
        "_vqa_source": None,  # internal marker
    })
    bdd_passthrough = []

    for entry in entries:
        sample = entry.get("sample", "")
        label  = entry.get("label", "")
        view   = entry.get("view", "")

        if sample.startswith("video"):
            bdd_passthrough.append(entry)
            continue

        key = (sample, label)
        dst = merged_map[key]
        dst["sample"] = sample
        dst["label"]  = label

        dst["image"].extend(entry.get("image", []))

        incoming_vqa = entry.get("vqa", []) or []
        if view == "overhead":
            if incoming_vqa:
                dst["vqa"] = list(incoming_vqa)
                dst["_vqa_source"] = "overhead"
            vqa_id = entry.get("vqa_id")
            if vqa_id and not dst["vqa_id"]:
                dst["vqa_id"] = vqa_id
        else:
            if not dst["vqa"] and incoming_vqa:
                dst["vqa"] = list(incoming_vqa)
                dst["_vqa_source"] = "vehicle"

    merged = []
    for _, e in merged_map.items():
        e["image"] = list(dict.fromkeys(e["image"]))
        e["vqa"] = _dedup_vqa_list(e.get("vqa", []))
        e["num_images"] = len(e["image"])
        e.pop("_vqa_source", None)
        merged.append(e)

    for entry in bdd_passthrough:
        imgs = entry.get("image", [])
        entry["num_images"] = len(imgs)
        merged.append(entry)

    return merged

def get_image_dimensions(image_paths):
    width_list, height_list = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            width_list.append(w)
            height_list.append(h)
        except:
            width_list.append(0)
            height_list.append(0)
    return width_list, height_list

def convert_to_jsonl(entries, processing_mode):
    output = []
    for idx, entry in enumerate(entries):
        sample, label, view = entry.get("sample", ""), entry.get("label", ""), entry.get("view", "")
        vqa_items = entry.get("vqa", [])
        images = entry.get("image", [])
        width_list, height_list = get_image_dimensions(images)
        is_wts = not sample.startswith("video")
        vqa_id = entry.get("vqa_id", None)

        conv = []
        for vqa in vqa_items:
            q = vqa.get("question", "")
            correct = vqa.get("correct", "")
            options = vqa.get("options", {
                k: vqa[k] for k in ['a', 'b', 'c', 'd'] if k in vqa
            })
            # For test sets, `correct` will be "", so `key` will also be ""
            key = correct.upper() if correct else ""

            image_block = "\n".join(["<image>"] * len(images))
            options_text = "\n".join(f"{k.upper()}. {v}" for k, v in sorted(options.items()))

            if processing_mode == 'evenly':
                prompt = (
                    f"{image_block}\n\n"
                    f"This accident scenario is in {label} phase.\n\n"
                    f"If red and blue boxes are present in the images, the red box highlights the pedestrian and the blue box highlights the vehicle.\n\n"
                    f"{q}\n{options_text}\n\n"
                    "Answer with the option's letter from the given choices directly."
                )
            else:
                view_text = (
                    "Images include both overhead views (showing full scene context) and vehicle views (from the assailant's perspective)."
                    if is_wts else
                    "Images are from the vehicle's front view (assailant's perspective)."
                )
                prompt = (
                    f"{image_block}\n\n"
                    f"This accident scenario is in {label} phase.\n\n"
                    f"{view_text}\n\n"
                    "If red and blue boxes are present, the red box highlights the pedestrian and the blue box highlights the vehicle.\n\n"
                    f"{q}\n{options_text}\n\n"
                    "Answer with the option's letter from the given choices directly."
                )

            conv.append({"from": "human", "value": prompt})
            conv.append({"from": "gpt", "value": key})

        meta = {"sample": sample, "label": label, "view": view}
        if vqa_id is not None:
            meta["vqa_id"] = vqa_id

        output.append({
            "id": idx,
            "image": images,
            "width_list": width_list,
            "height_list": height_list,
            "conversations": conv,
            "meta": meta
        })
    return output

def extract_gender(text: str) -> str:
    if not text: return "NULL"
    t = text.lower()
    if re.search(r"\b(female|woman|women)\b", t): return "Female"
    if re.search(r"\b(male|man|men)\b", t): return "Male"
    return "NULL"

def extract_speed(caption: str) -> str:
    if not caption: return "NULL"
    m = re.search(r"(\d+)\s*km/?h", caption)
    if m:
        speed = int(m.group(1))
        if speed in [0, 5, 10, 15, 20, 25, 30]:
            return f"{speed} km/h"
    return "NULL"

def append_two_caption_QAs_train(entries):
    """Appends gender/speed QAs with extracted answers for TRAIN mode."""
    valid_speeds = [f"{s} km/h" for s in [0, 5, 10, 15, 20, 25, 30]]
    for e in entries:
        e.setdefault("vqa", [])
        ped_cap = e.get("caption_pedestrian", "") or ""
        veh_cap = e.get("caption_vehicle", "") or ""

        # Gender QA
        gender = extract_gender(ped_cap or veh_cap)
        if gender == "NULL": gender = "Male"
        opposite = "Male" if gender == "Female" else "Female"
        opts = [gender, opposite]
        random.shuffle(opts)
        correct_letter = "a" if opts[0] == gender else "b"
        e["vqa"].append({
            "question": "What is the gender of the Pedestrian?",
            "options": {"a": opts[0], "b": opts[1]},
            "correct": correct_letter
        })

        # Speed QA
        speed = extract_speed(veh_cap)
        if speed == "NULL": speed = "0 km/h"
        distractor = random.choice([s for s in valid_speeds if s != speed] or ["5 km/h"])
        opts = [speed, distractor]
        random.shuffle(opts)
        correct_letter = "a" if opts[0] == speed else "b"
        e["vqa"].append({
            "question": "What is the speed of the Assailant vehicle?",
            "options": {"a": opts[0], "b": opts[1]},
            "correct": correct_letter
        })
    return entries

def append_two_caption_QAs_test(entries):
    """Appends placeholder gender/speed questions with full options for TEST mode."""
    # Define the standardized options once
    gender_options = {"a": "Male", "b": "Female"}
    speed_options = {chr(97 + i): f"{s} km/h" for i, s in enumerate(range(0, 31, 5))}

    for e in entries:
        e.setdefault("vqa", [])
        
        # Gender QA with standardized options
        e["vqa"].append({
            "question": "What is the gender of the Pedestrian?",
            "options": gender_options,
            "correct": ""
        })

        # Speed QA with all possible options for a comprehensive evaluation
        e["vqa"].append({
            "question": "What is the speed of the Assailant vehicle?",
            "options": speed_options,
            "correct": ""
        })
    return entries

def _normalize_str(x):
    return (str(x) if x is not None else "").strip().lower()

def _vqa_options_dict(vqa):
    if isinstance(vqa.get("options"), dict):
        opts = vqa["options"]
    else:
        opts = {k: vqa[k] for k in ["a", "b", "c", "d"] if k in vqa}
    return { _normalize_str(k): _normalize_str(v) for k, v in opts.items() }

def _vqa_signature(vqa):
    q = _normalize_str(vqa.get("question", ""))
    opts_sorted = tuple(sorted(_vqa_options_dict(vqa).items()))
    correct = _normalize_str(vqa.get("correct", ""))
    return (q, opts_sorted, correct)

def _dedup_vqa_list(vqa_list):
    seen, out = set(), []
    for v in vqa_list or []:
        sig = _vqa_signature(v)
        if sig in seen: continue
        seen.add(sig)
        out.append(v)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wts-json', required=True)
    parser.add_argument('--bdd-json', required=True)
    parser.add_argument('--view-csv', required=True)
    parser.add_argument('--jsonl-out', required=True)
    parser.add_argument('--pretty-json-out', required=True)
    parser.add_argument('--log-dir', default="logs")
    parser.add_argument('--dataset-type', choices=['all', 'wts', 'bdd'], default='all',
                        help="Choose to keep 'wts' only, 'bdd' only, or 'all' samples")
    parser.add_argument('--processing-mode', choices=['evenly', 'centered'], default='centered',
                        help="Processing mode: 'centered' appends QAs first, 'evenly' removes empty samples first.")
    parser.add_argument('--is-test-set', action='store_true',
                        help="Process as a test set. This will generate questions without answers for appended QAs.")
    args = parser.parse_args()

    random.seed(42)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    kept_log = Path(args.log_dir) / "kept_samples.txt"
    removed_log = Path(args.log_dir) / "removed_samples.txt"
    cam_map = Path(args.log_dir) / "sample_to_camera.json"

    print("[1] Merging both the dataset's input files...")
    merged = merge_json_files(args.wts_json, args.bdd_json)
    print(f"    Total merged entries: {len(merged)}")

    data_to_process = merged
    print("[2] Filtering WTS overhead views based on best view scenarios...")
    views = load_valid_views(args.view_csv)
    print(f"    Valid view entries loaded: {len(views)}")
    data_to_process = filter_wts_overhead(merged, views, kept_log, removed_log, cam_map)
    print(f"    Entries after WTS overhead filtering: {len(data_to_process)}")

    print("[3] Merging WTS overhead + vehicle view samples...")
    merged_views = merge_overhead_and_vehicle(data_to_process)
    print(f"    Total after overhead + vehicle merge: {len(merged_views)}")

    if args.dataset_type == 'wts':
        merged_views = [entry for entry in merged_views if not entry.get("sample", "").startswith("video")]
        print(f"[✓] Filtered WTS-only samples: {len(merged_views)}")
    elif args.dataset_type == 'bdd':
        merged_views = [entry for entry in merged_views if entry.get("sample", "").startswith("video")]
        print(f"[✓] Filtered BDD-only samples: {len(merged_views)}")
    else:
        print("[✓] Keeping all WTS + BDD samples")

    should_append_qas = not (args.processing_mode == 'evenly' and args.dataset_type == 'bdd')

    if args.processing_mode == 'centered':
        print("[MODE: CENTERED] Appending QAs before any filtering.")
        if should_append_qas:
            print("[3b] Appending 2 caption-based QAs (gender + speed)...")
            merged_views = append_two_caption_QAs(merged_views, is_test_set=args.is_test_set)
            print("Caption-based QAs appended.")
    elif args.processing_mode == 'evenly':
        print("[MODE: EVENLY] Removing empty samples before appending QAs.")
        print("[3b] Removing samples with empty VQA lists...")
        original_count = len(merged_views)
        merged_views = [entry for entry in merged_views if entry.get("vqa")]
        print(f"    Removed {original_count - len(merged_views)} samples. {len(merged_views)} remain.")
        
        if should_append_qas:
            print("[3c] Appending 2 caption-based QAs (gender + speed)...")
            if args.is_test_set:
                print("    -> Using TEST mode function with full options.")
                merged_views = append_two_caption_QAs_test(merged_views)
            else:
                print("    -> Using TRAIN mode function with extracted answers.")
                merged_views = append_two_caption_QAs_train(merged_views)
            
            print("Caption-based QAs appended.")
        else:
            print("[3c] Skipping append of caption-based QAs for evenly/bdd combination.")


    print("[4] Converting to Finetuning JSONL format...")
    converted = convert_to_jsonl(merged_views, args.processing_mode)
    print(f"JSONL entries before filtering: {len(converted)}")

    final_output = []
    if args.processing_mode == 'centered':
        final_output = converted
        print("[6] Skipping removal of empty conversations for 'centered' mode.")
    elif args.processing_mode == 'evenly':
        print("[6] Removing samples with empty conversations...")
        final_output = [entry for entry in converted if entry.get("conversations")]
        print(f"JSONL entries after filtering: {len(final_output)}")

    with open(args.jsonl_out, 'w') as f:
        for entry in final_output:
            f.write(json.dumps(entry) + "\n")

    with open(args.pretty_json_out, 'w') as f:
        json.dump(final_output, f, indent=2)

    print("Done. Prepared and Saved in Finetuning Format.")

if __name__ == "__main__":
    main()
