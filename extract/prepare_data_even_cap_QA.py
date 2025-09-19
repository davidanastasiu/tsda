import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm


def extract_gender(text):
    """Extracts gender from caption text, defaulting to Male if ambiguous."""
    if not text: return "Male"
    text = text.lower()
    if any(word in text for word in ["female", "woman", "women"]):
        return "Female"
    if any(word in text for word in ["male", "man", "men"]):
        return "Male"
    return "Male"  


def extract_speed(caption):
    """Extracts vehicle speed from caption text, defaulting to 0 km/h."""
    if not caption: return "0 km/h"
    match = re.search(r"(\d+)\s*km/?h", caption)
    if match:
        speed = int(match.group(1))
        if speed in [0, 5, 10, 15, 20, 25, 30]:
            return f"{speed} km/h"
    return "0 km/h"  # Default fallback


def merge_json_files(wts_path, bdd_path):
    """Merges two JSON files into a single list."""
    with open(wts_path, 'r') as f:
        wts_data = json.load(f)
    with open(bdd_path, 'r') as f:
        bdd_data = json.load(f)
    return [dict(entry, id=idx) for idx, entry in enumerate(wts_data + bdd_data)]


def merge_wts_views(entries):
    """Merges WTS overhead and vehicle views into a single entry per scene."""
    merged_map = defaultdict(lambda: {
        "image": [], "vqa": [], "caption_pedestrian": "", "caption_vehicle": ""
    })
    bdd_passthrough = []
    for entry in entries:
        sample = entry.get("sample", "")
        if sample.startswith("video"):
            bdd_passthrough.append(entry)
            continue
        key = (sample, entry.get("label"))
        dst = merged_map[key]
        dst.update({"sample": sample, "label": entry.get("label"), "view": "merged"})
        dst["image"].extend(entry.get("image", []))
        dst["image"] = sorted(list(set(dst["image"])))
        if not dst["vqa"] and entry.get("vqa"):
            dst["vqa"] = entry["vqa"]
        if entry.get("caption_pedestrian"):
            dst["caption_pedestrian"] = entry["caption_pedestrian"]
        if entry.get("caption_vehicle"):
            dst["caption_vehicle"] = entry["caption_vehicle"]
    return list(merged_map.values()) + bdd_passthrough


def get_image_dimensions(image_paths):
    """Gets the dimensions of a list of images, returning 0 for errors."""
    width_list, height_list = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                w, h = img.size
            width_list.append(w)
            height_list.append(h)
        except (IOError, FileNotFoundError):
            width_list.append(0)
            height_list.append(0)
    return width_list, height_list


def format_vqa_to_qa_block(vqa_list):
    """Converts a list of VQA dicts into a Q:/A: formatted string block."""
    qa_pairs, seen_questions = [], set()
    for vqa in vqa_list:
        question = vqa.get("question", "").strip()
        if not question or question.lower() in seen_questions:
            continue
        seen_questions.add(question.lower())
        correct_key = vqa.get("correct", "")
        options = vqa.get("options", {})
        answer = options.get(correct_key, "").strip()
        if not answer:
            continue
        formatted_question = question[0].upper() + question[1:]
        qa_pairs.append(f"Q: {formatted_question}\nA: {answer}")
    return "\n".join(qa_pairs)


def generate_even_qa_file(entries, out_path):
    """Generates the single combined file for the even_QA format."""
    output_data = []
    print("Generating even_QA file...")
    for idx, entry in enumerate(tqdm(entries)):
        cap_ped = entry.get("caption_pedestrian", "").strip()
        cap_veh = entry.get("caption_vehicle", "").strip()
        
       
        if not cap_ped or not cap_veh:
            continue

        vqa_list = list(entry.get("vqa", []))

        gender = extract_gender(cap_ped)
        speed = extract_speed(cap_veh)
        vqa_list.append({"question": "What is the gender of Pedestrian?", "options": {"a": gender}, "correct": "a"})
        vqa_list.append({"question": "What is speed of Assailant Vehicle?", "options": {"a": speed}, "correct": "a"})

        images = entry.get("image", [])
        width_list, height_list = get_image_dimensions(images)
        image_block = "\n".join(["<image>"] * len(images))
        qa_block = format_vqa_to_qa_block(vqa_list)

        prompt = (
            f'{image_block}\n\nYou are given multiple images for "{entry.get("label", "default")}" phase of an accident scenario.\n\n'
            "If red and blue boxes are present in the images, the red box highlights the pedestrian and the blue box highlights the vehicle.\n\n"
            "Carefully analyze the scene using information from all available views. Focus on spatial positioning, visibility, and interactions between pedestrians and vehicles.\n\n"
            "Generate two detailed captions in the following format:\n"
            "Ped_caption: <Insert detailed pedestrian description here>\n"
            "Veh_caption: <Insert detailed vehicle description here>\n"
            "Before generating the final descriptions, consider the following annotated information:\n\n"
            f'{qa_block}'
        )
        
        gpt_response = f"Ped_caption: {cap_ped}\nVeh_caption: {cap_veh}"

        output_data.append({
            "image": images,
            "width_list": width_list,
            "height_list": height_list,
            "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": gpt_response}],
            "meta": {"sample": entry.get("sample", ""), "label": entry.get("label", ""), "view": "merged"}
        })

    with open(out_path, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(output_data)} entries to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate the even_QA finetuning file.")
    parser.add_argument('--wts-json', required=True, help="Path to the input WTS JSON file.")
    parser.add_argument('--bdd-json', required=True, help="Path to the input BDD JSON file.")
    parser.add_argument('--out-jsonl', required=True, help="Path for the final output JSONL file.")
    args = parser.parse_args()

    random.seed(42)

    print("[1/3] Merging WTS and BDD input files...")
    merged_data = merge_json_files(args.wts_json, args.bdd_json)
    print(f"Total entries: {len(merged_data)}")

    print("\n[2/3] Merging WTS overhead and vehicle views (no best view filtering)...")
    final_entries = merge_wts_views(merged_data)
    print(f"Entries after merging: {len(final_entries)}")

    print("\n[3/3] Generating the final even_QA file...")
    generate_even_qa_file(final_entries, args.out_jsonl)

    print("\n✨ All tasks completed successfully!")


if __name__ == "__main__":
    main()