import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import copy


    
QA_TO_FACT_MAP = {
    "what is the age group of the pedestrian?": {"template": "The pedestrian is in the {} age group."},
    "what is the height of the pedestrian?": {"template": "The pedestrian is {} tall."},
    "is pedestrian wearing a hat?": {"Yes": "The pedestrian is wearing a hat.", "No": "The pedestrian is not wearing a hat."},
    "what is color of pedestrian's hat?": {"template": "The pedestrian's hat is {}."},
    "what pedestrian is wearing on upper body?": {"template": "The pedestrian is wearing a {} on the upper body."},
    "what is the color of pedestrian's upper body clothing?": {"template": "The color of pedestrian's upper body clothing is {}."},
    "what pedestrian is wearing on lower body?": {"template": "The pedestrian is wearing {} on the lower body."},
    "what is the color of pedestrian's lower body clothing?": {"template": "The color of pedestrian's lower body clothing is {}."},

    "what is weather in the scenario?": {"template": "The weather is {}."},
    "what is the brightness level in the scene?": {"template": "The brightness of scene is {}."},
    "what are road surface conditions?": {"template": "The road surface conditions are {}."},
    "what is the road inclination in the scene?": {"template": "The road is {}."},
    "what is surface type of the road?": {"template": "The road surface is made of {}."},
    "what is the volume of the traffic in the scene?": {"template": "The traffic volume is {}."},
    "what is the type of the road?": {"template": "The road is a {}."},
    "how many lanes are there?": {"template": "The road is with {}."},

    "where is the sidewalk in the scene?": {
        "Both sides": "There are sidewalks on both sides of the road.",
        "Not both sides": "Sidewalks are not present on both sides of the road.",
        "Only on the left": "There is a sidewalk only on the left side of the road.",
        "Only on the right": "There is a sidewalk only on the right side of the road."
    },
    "where is the roadside strip in the scene?": {
        "Both sides": "There are roadside strips on both sides of the road.",
        "Not both sides": "Roadside strips are not present on both sides of the road.",
        "Only on the left": "There is a roadside strip only on the left side of the road.",
        "Only on the right": "There is a roadside strip only on the right side of the road."
    },
    "are there street lights in the scene?": {"Yes": "There are street lights in the scene.", "No": "There are no street lights in the scene."},

    "what is the position of the obstacle in the scene?": {"template": "The obstacle is positioned {}."},
    "what is the height of obstacle in the scene?": {"template": "The obstacle is {} high."},
    "what is the width of obstacle in the scene?": {"template": "The obstacle is {} wide."},

    "what is the orientation of the pedestrian's body?": {"template": "The pedestrian is oriented {}."},
    "what is the position of the pedestrian relative to the vehicle?": {"template": "The pedestrian is {} relative to the vehicle."},
    "what is relative distance of pedestrian from vehicle?": {"template": "The pedestrian is at a {} distance from the vehicle."},

    "what is the pedestrian's line of sight?": {"template": "Pedestrian was closely watching the {}."},
    "what is the pedestrian's visual status?": {"template": "The pedestrian is {}."},

    "what is the pedestrian's awareness regarding vehicle?": {
        "Notices the vehicle": "The pedestrian appears to be aware of the vehicle.",
        "Unaware of the vehicle": "The pedestrian does not appear to be aware of the vehicle.",
        "Cannot be determined": "The pedestrian's awareness regarding the vehicle cannot be determined."
    },

    
    "what is the pedestrian's action?": {"template": "The pedestrian is {}."},


    "what is the position of the vehicle relative to the pedestrian?": {"template": "The vehicle is positioned {} relative to the pedestrian."},
    "what is relative distance of vehicle from pedestrian?": {"template": "The vehicle is at a {} distance from the pedestrian."},
    "what is vehicle's field of view?": {
        "Pedestrian is visible": "The pedestrian is visible in the vehicle's field of view.",
        "Pedestrian is not visible": "The pedestrian is not visible in the vehicle's field of view."
    },
    "what is the action taken by vehicle?": {"template": "The vehicle is {}."},

    
    "what is the gender of the pedestrian?": {"template": "The pedestrian is {}."},
    "what is the speed of the assailant vehicle?": {"template": "The assailant vehicle is moving at {}."},

    "what is the pedestrian's direction of travel?": {"template": "The pedestrian's direction of travel is {}."},
    "what is pedestrian's speed?": {"template": "The pedestrian is moving {}."},
    "what is the fine-grained action taken by the pedestrian?": {"template": "The pedestrian is {}."},
    "is the pedestrian wearing glasses?": {"Yes": "The pedestrian is wearing glasses.", "No": "The pedestrian is not wearing glasses."},

    "what is the setting of this scenario?": {"template": "The scenario is set in a {} environment."},
    "what is the formation of the road?": {"template": "The road is {}."},
    "is there guardrail around the road?": {"Yes": "There is a guardrail around the road.", "No": "There is no guardrail around the road."},
    "is the pedestrian holding a walking cane?": {"Yes": "The pedestrian is holding a walking cane.", "No": "The pedestrian is not holding a walking cane."}
}

LABELS_int = ["action", "judgement", "avoidance", "precognition", "prerecognition"]
LABELS_ext = ["prerecognition", "recognition", "judgement", "action", "avoidance"]

def build_caption_lookup(entries):
    """Builds a lookup table to find captions by sample and label."""
    lookup = defaultdict(dict)
    for entry in entries:
        sample_id = entry.get("sample")
        label = entry.get("label")
        if sample_id and label:
            # Store captions only if they exist
            ped_cap = entry.get("caption_pedestrian")
            veh_cap = entry.get("caption_vehicle")
            if ped_cap or veh_cap:
                lookup[sample_id][label] = {
                    "ped": ped_cap,
                    "veh": veh_cap
                }
    return lookup

def _normalize_label(label):

    try:
        return LABELS_ext[LABELS_int.index(label)]
    except ValueError:
        return label

def map_captions_by_fetching(entries, lookup):
    remapped_entries = []
    for entry in entries:
        new_entry = copy.deepcopy(entry)
        sample_id = new_entry.get("sample", "")

        
        if not sample_id.startswith("video"):
            original_label = new_entry.get("label")
            target_caption_label = _normalize_label(original_label)

            if target_caption_label and sample_id in lookup and target_caption_label in lookup[sample_id]:
                correct_captions = lookup[sample_id][target_caption_label]
                new_entry["caption_pedestrian"] = correct_captions.get("ped", "")
                new_entry["caption_vehicle"] = correct_captions.get("veh", "")
        
        remapped_entries.append(new_entry)
    return remapped_entries

def extract_gender(text):
    """Extracts gender from caption text."""
    if not text: return "Male"
    text = text.lower()
    if any(word in text for word in ["female", "woman", "women"]): return "Female"
    return "Male"

def extract_speed(caption):
    """Extracts vehicle speed from caption text."""
    if not caption: return "0 km/h"
    match = re.search(r"(\d+)\s*km/?h", caption)
    if match and int(match.group(1)) in [0, 5, 10, 15, 20, 25, 30]:
        return f"{match.group(1)} km/h"
    return "0 km/h"

def append_caption_based_vqa(entry):
    """Appends gender and speed VQA to an entry."""
    ped_caption = entry.get("caption_pedestrian", "")
    veh_caption = entry.get("caption_vehicle", "")
    if "vqa" not in entry: entry["vqa"] = []
    
    gender = extract_gender(ped_caption)
    entry["vqa"].append({
        "question": "what is the gender of the pedestrian?",
        "options": {"a": gender, "b": "Male" if gender == "Female" else "Female"},
        "correct": "a"
    })
    
    speed = extract_speed(veh_caption)
    valid_speeds = [f"{s} km/h" for s in [0, 5, 10, 15, 20, 25, 30]]
    distractor = random.choice([s for s in valid_speeds if s != speed] or ["5 km/h"])
    opts = [speed, distractor]; random.shuffle(opts)
    entry["vqa"].append({
        "question": "what is the speed of the assailant vehicle?",
        "options": {"a": opts[0], "b": opts[1]},
        "correct": "a" if opts[0] == speed else "b"
    })
    return entry

def merge_json_files(wts_path, bdd_path):
    """Merges two JSON files."""
    with open(wts_path, 'r') as f: wts_data = json.load(f)
    with open(bdd_path, 'r') as f: bdd_data = json.load(f)
    return [dict(entry, id=idx) for idx, entry in enumerate(wts_data + bdd_data)]

def load_valid_views(csv_path):
    """Loads valid camera views from a CSV."""
    df = pd.read_csv(csv_path)
    return {row["Scene"]: [Path(row[col]).stem for col in ["Viewpoint1", "Viewpoint2", "Viewpoint3", "Viewpoint4"] if pd.notna(row[col])] for _, row in df.iterrows()}

def filter_wts_overhead(entries, valid_views):
    """Keeps images from all specified best cameras for WTS overhead views."""
    filtered_entries = []
    print("Filtering WTS overhead views...")
    for entry in tqdm(entries):
        sample, view = entry.get("sample", ""), entry.get("view", "")
        if not sample.startswith("video") and view == "overhead":
            best_views = valid_views.get(sample)
            if best_views:
                kept_images = [img for img in entry.get("image", []) if any(cam in img for cam in best_views)]
                if kept_images: entry["image"] = kept_images
        filtered_entries.append(entry)
    return filtered_entries

def merge_wts_views(entries):
    """Merges WTS overhead and vehicle views."""
    merged_map = defaultdict(lambda: {"image": [], "vqa": [], "caption_pedestrian": "", "caption_vehicle": ""})
    bdd_passthrough = []
    for entry in entries:
        if entry.get("sample", "").startswith("video"):
            bdd_passthrough.append(entry)
            continue
        key = (entry.get("sample"), entry.get("label"))
        dst = merged_map[key]
        dst.update({"sample": entry.get("sample"), "label": entry.get("label"), "view": "merged"})
        dst["image"].extend(entry.get("image", []))
        dst["image"] = sorted(list(set(dst["image"])))
        if not dst["vqa"] and entry.get("vqa"):
            dst["vqa"] = entry.get("vqa")
        if not dst["caption_pedestrian"] and entry.get("caption_pedestrian"):
            dst["caption_pedestrian"] = entry.get("caption_pedestrian")
        if not dst["caption_vehicle"] and entry.get("caption_vehicle"):
            dst["caption_vehicle"] = entry.get("caption_vehicle")
    return list(merged_map.values()) + bdd_passthrough

def get_image_dimensions(image_paths):
    """Gets dimensions for a list of images."""
    w_list, h_list = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img: w, h = img.size
            w_list.append(w); h_list.append(h)
        except (IOError, FileNotFoundError):
            w_list.append(0); h_list.append(0)
    return w_list, h_list

def convert_vqa_to_facts(vqa_list):
    """Converts a VQA list into a list of fact strings."""
    facts, seen_questions = [], set()
    for vqa in vqa_list:
        question = vqa.get("question", "").lower().strip()
        if question in seen_questions: continue
        seen_questions.add(question)
        answer = vqa.get("options", {}).get(vqa.get("correct", ""), "")
        if not answer: continue
        mapping = QA_TO_FACT_MAP.get(question)
        if not mapping: continue
        if "template" in mapping:
            facts.append(mapping["template"].format(answer))
        else:
            for map_key, fact_string in mapping.items():
                if map_key.lower() == answer.lower():
                    facts.append(fact_string)
                    break
    return facts

def generate_combined_caption_file(entries, out_path):
    """Generates the final combined caption file."""
    output_data = []
    print("Generating combined caption file...")
    for idx, entry in enumerate(tqdm(entries)):
        facts = convert_vqa_to_facts(entry.get("vqa", []))
        images = entry.get("image", [])
        w, h = get_image_dimensions(images)
        image_block = "\n".join(["<image>"] * len(images))
        facts_block = "\n".join(facts)
        
        prompt_template = (
            f'{image_block}\n\nYou are given multiple images for the "{entry.get("label", "")}" phase of an accident scenario.\n\n'
            "Generate only the {} description using all the available visual cues and associated facts.\n\n"
            f'{facts_block}'
        )
        output_data.append({
            "image": images, "width_list": w, "height_list": h,
            "conversations": [
                {"from": "human", "value": prompt_template.format("pedestrian")},
                {"from": "gpt", "value": entry.get("caption_pedestrian", "").strip()},
                {"from": "human", "value": prompt_template.format("vehicle")},
                {"from": "gpt", "value": entry.get("caption_vehicle", "").strip()}
            ],
            "meta": {"sample": entry.get("sample", ""), "label": entry.get("label", ""), "view": "merged"}
        })

    with open(out_path, 'w') as f:
        for item in output_data: f.write(json.dumps(item) + "\n")
    print(f"✅ Saved {len(output_data)} entries to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate remapped captioning files.")
    parser.add_argument('--wts-json', required=True)
    parser.add_argument('--bdd-json', required=True)
    parser.add_argument('--view-csv', required=True)
    parser.add_argument('--out-jsonl', required=True)
    args = parser.parse_args()

    random.seed(42)

    print("[1/6] Merging WTS and BDD input files...")
    merged_data = merge_json_files(args.wts_json, args.bdd_json)
    
    print("\n[2/6] Building caption lookup table...")
    caption_lookup = build_caption_lookup(merged_data)
    mapped_data = map_captions_by_fetching(merged_data, caption_lookup)
    
    print("\n[3/6] Filtering WTS overhead views...")
    valid_views = load_valid_views(args.view_csv)
    filtered_data = filter_wts_overhead(mapped_data, valid_views)

    print("\n[4/6] Merging WTS views...")
    merged_views = merge_wts_views(filtered_data)
    
    final_entries = [e for e in merged_views if e.get("caption_pedestrian", "").strip() and e.get("caption_vehicle", "").strip()]
    print(f"    Entries after filtering empty captions: {len(final_entries)}")

    print("\n[5/6] Appending gender and speed VQA...")
    augmented_entries = [append_caption_based_vqa(entry) for entry in tqdm(final_entries)]

    print("\n[6/6] Generating final caption file...")
    generate_combined_caption_file(augmented_entries, args.out_jsonl)

    print("\n✨ All tasks completed successfully!")

if __name__ == "__main__":
    main()