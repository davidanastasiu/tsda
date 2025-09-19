import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
from PIL import Image


ENV_QUESTIONS = {
    "what is the age group of the pedestrian?", "what is the height of the pedestrian?", "what pedestrian is wearing on upper body?",
    "what is the color of pedestrian's upper body clothing?", "what pedestrian is wearing on lower body?", "what is the color of pedestrian's lower body clothing?",
    "what is weather in the scenario?", "what is the brightness level in the scene?", "what are road surface conditions?", "what is the road inclination in the scene?",
    "what is surface type of the road?", "what is the volume of the traffic in the scene?", "what is the type of the road?", "how many lanes are there?",
    "where is the sidewalk in the scene?", "where is the roadside strip in the scene?", "are there street lights in the scene?", "what is the position of the obstacle in the scene?",
    "what is the height of obstacle in the scene?", "what is the width of obstacle in the scene?", "is pedestrian wearning a hat?", "what is color of pedestrian's hat?",
    "is the pedestrian wearing glasses?", "what is the formation of the road?", "is the pedestrian holding a waling cane?", "what is the setting of this scenario?",
    "is there guardrail around the road?"
}
PED_QUESTIONS = {
    "what is the orientation of the pedestrian's body?", "what is the position of the pedestrian relative to the vehicle?", "what is relative distance of pedestrian from vehicle?",
    "what is the pedestrian's line of sight?", "what is the pedestrian's visual status?", "what is the pedestrian's direction of travel?", "what is the pedestrian's action?",
    "what is pedestrian's speed?", "what is the pedestrian's awareness regarding vehicle?", "what is the fine-grained action taken by the pedestrian?"
}
VEH_QUESTIONS = {
    "what is vehicle's field of view?", "what is the action taken by vehicle?", "what is the position of the vehicle relative to the pedestrian?",
    "what is relative distance of vehicle from pedestrian?"
}

QA_TO_FACT_MAP = {
    "what is the age group of the pedestrian": {"template": "The pedestrian is in the {} age group."},
    "what is the height of the pedestrian": {"template": "The pedestrian is {} tall."},
    "is pedestrian wearning a hat": {"Yes": "The pedestrian is wearing a hat.", "No": "The pedestrian is not wearing a hat."},
    "what is color of pedestrian's hat": {"template": "The pedestrian's hat is {}."},
    "what pedestrian is wearing on upper body": {"template": "The pedestrian is wearing a {} on the upper body."},
    "what is the color of pedestrian's upper body clothing": {"template": "The color of pedestrian's upper body clothing is {}."},
    "what pedestrian is wearing on lower body": {"template": "The pedestrian is wearing {} on the lower body."},
    "what is the color of pedestrian's lower body clothing": {"template": "The color of pedestrian's lower body clothing is {}."},

    "what is weather in the scenario": {"template": "The weather is {}."},
    "what is the brightness level in the scene": {"template": "The brightness of scene is {}."},
    "what are road surface conditions": {"template": "The road surface conditions are {}."},
    "what is the road inclination in the scene": {"template": "The road is {}."},
    "what is surface type of the road": {"template": "The road surface is made of {}."},
    "what is the volume of the traffic in the scene": {"template": "The traffic volume is {}."},
    "what is the type of the road": {"template": "The road is a {}."},
    "how many lanes are there": {"template": "The road is with {}."},

    "where is the sidewalk in the scene": {
        "Both sides": "There are sidewalks on both sides of the road.",
        "Not both sides": "Sidewalks are not present on both sides of the road.",
        "Only on the left": "There is a sidewalk only on the left side of the road.",
        "Only on the right": "There is a sidewalk only on the right side of the road."
    },
    "where is the roadside strip in the scene": {
        "Both sides": "There are roadside strips on both sides of the road.",
        "Not both sides": "Roadside strips are not present on both sides of the road.",
        "Only on the left": "There is a roadside strip only on the left side of the road.",
        "Only on the right": "There is a roadside strip only on the right side of the road."
    },
    "are there street lights in the scene": {"Yes": "There are street lights in the scene.", "No": "There are no street lights in the scene."},

    "what is the position of the obstacle in the scene": {"template": "The obstacle is positioned {}."},
    "what is the height of obstacle in the scene": {"template": "The obstacle is {} high."},
    "what is the width of obstacle in the scene": {"template": "The obstacle is {} wide."},

    "what is the orientation of the pedestrian's body": {"template": "The pedestrian is oriented {}."},
    "what is the position of the pedestrian relative to the vehicle": {"template": "The pedestrian is {} relative to the vehicle."},
    "what is relative distance of pedestrian from vehicle": {"template": "The pedestrian is at a {} distance from the vehicle."},

    "what is the pedestrian's line of sight": {"template": "Pedestrian's line of sight was focused on {}."},
    "what is the pedestrian's visual status": {"template": "The pedestrian is {}."},

    "what is the pedestrian's awareness regarding vehicle": {
        "Notices the vehicle": "The pedestrian appears to be aware of the vehicle.",
        "Unaware of the vehicle": "The pedestrian does not appear to be aware of the vehicle.",
        "Cannot be determined": "The pedestrian's awareness regarding the vehicle cannot be determined."
    },


    "what is the pedestrian's action": {"template": "The pedestrian is {}."},

    "what is the position of the vehicle relative to the pedestrian": {"template": "The vehicle is positioned {} relative to the pedestrian."},
    "what is relative distance of vehicle from pedestrian": {"template": "The vehicle is at a {} distance from the pedestrian."},
    "what is vehicle's field of view": {
        "Pedestrian is visible": "The pedestrian is visible in the vehicle's field of view.",
        "Pedestrian is not visible": "The pedestrian is not visible in the vehicle's field of view."
    },
    "what is the action taken by vehicle": {"template": "The vehicle is {}."},


    "what is the gender of pedestrian": {"template": "The pedestrian is {}."},
    "what is speed of assailant vehicle": {"template": "The assailant vehicle is moving at {}."},

    "what is the pedestrian's direction of travel": {"template": "The pedestrian's direction of travel is {}."},
    "what is pedestrian's speed": {"template": "The pedestrian is moving {}."},
    "what is the fine-grained action taken by the pedestrian": {"template": "The pedestrian is {}."},
    "is the pedestrian wearing glasses": {"Yes": "The pedestrian is wearing glasses.", "No": "The pedestrian is not wearing glasses."},

    "what is the setting of this scenario": {"template": "The scenario is set in a {} environment."},
    "what is the formation of the road": {"template": "The road is {}."},
    "is there guardrail around the road": {"Yes": "There is a guardrail around the road.", "No": "There is no guardrail around the road."},
    "is the pedestrian holding a waling cane": {"Yes": "The pedestrian is holding a walking cane.", "No": "The pedestrian is not holding a walking cane."}
}


def parse_complex_question(question_text):
    """Extracts the simple question and options from the complex VQA string."""
    question_pattern = re.compile(r'\n\n(What.*?)\?', re.DOTALL)
    options_pattern = re.compile(r'\n([A-G])\. (.*?)(?=\n[A-G]\.|\n\nAnswer)', re.DOTALL)
    
    question_match = question_pattern.search(question_text)
    simple_question = ""
    if question_match:
        simple_question = question_match.group(1).lower().strip()

    options_matches = options_pattern.findall(question_text)
    options = {k.lower(): v.strip() for k, v in options_matches}
    
    return simple_question, options

def load_multiagent_answers(path):
    """Loads answers from the multiagent file, skipping gender/speed."""
    answers_map = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("meta", {})
            key = (meta.get("sample"), meta.get("label"))
            if not all(key): continue

            for pred in item.get("predictions", []):
                q = pred.get("question", "").lower().strip()
                if "gender of pedestrian" not in q and "speed of assailant vehicle" not in q:
                    answers_map[key][q] = pred.get("answer", "")
    return answers_map

def load_gender_speed_answers(ref_entries):
    """Loads ONLY gender and speed answers from the pre-loaded full reference data."""
    lookup = defaultdict(dict)
    for item in ref_entries:
        meta = item.get("meta", {})
        key = (meta.get("sample"), meta.get("label"))
        if not all(key): continue

        for pred in item.get("predictions", []):
            simple_question, options = parse_complex_question(pred.get("question", ""))
            if "gender of pedestrian" in simple_question or "speed of assailant vehicle" in simple_question:
                answer_letter = pred.get("answer", "")
                answer_text = options.get(answer_letter.lower())
                if answer_text:
                    lookup[key][simple_question] = answer_text
    return lookup

def build_vqa_options_lookup(entries, ref_entries):
    """Builds a complete VQA options lookup from test data and the full reference file."""
    lookup = defaultdict(dict)
    for entry in entries: # From wts_test.json / bdd_test.json
        key = (entry.get("sample"), entry.get("label"))
        if all(key) and entry.get("vqa"):
            for qa_pair in entry["vqa"]:
                question = qa_pair.get("question", "").lower().strip()
                if question:
                    options = {k: v for k, v in qa_pair.items() if k in ['a', 'b', 'c', 'd', 'e', 'f', 'g']}
                    lookup[key][question] = options
    
    for entry in ref_entries: # From test_full_ref.jsonl
        key = (entry.get("meta", {}).get("sample"), entry.get("meta", {}).get("label"))
        if all(key):
            for pred in entry.get("predictions", []):
                simple_question, options = parse_complex_question(pred.get("question", ""))
                if simple_question and options and simple_question not in lookup[key]:
                    lookup[key][simple_question] = options
    return lookup

# --- Core Logic Functions ---

def resolve_all_answers(multiagent_answers, gender_speed_answers, options_lookup):
    """Combines answers from all sources into a final dictionary of {question: answer_text}."""
    final_answers = defaultdict(dict)
    for key, qa_pairs in multiagent_answers.items():
        for q, ans_letter in qa_pairs.items():
            options = options_lookup.get(key, {}).get(q, {})
            ans_text = options.get(ans_letter.lower())
            if ans_text:
                final_answers[key][q] = ans_text
    
    for key, qa_pairs in gender_speed_answers.items():
        final_answers[key].update(qa_pairs)
        
    return final_answers


def merge_json_files(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_valid_views(csv_path):
    df = pd.read_csv(csv_path)
    return {row["Scene"]: [Path(row[col]).stem for col in ["Viewpoint1", "Viewpoint2", "Viewpoint3", "Viewpoint4"] if pd.notna(row[col])] for _, row in df.iterrows()}

def filter_wts_overhead(entries, valid_views):
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
    merged_map = defaultdict(lambda: {"image": []})
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
    return list(merged_map.values()) + bdd_passthrough

def get_image_dimensions(image_paths):
    w_list, h_list = [], []
    for path in image_paths:
        try:
            with Image.open(path) as img: w, h = img.size
            w_list.append(w); h_list.append(h)
        except (IOError, FileNotFoundError):
            w_list.append(0); h_list.append(0)
    return w_list, h_list

def convert_answers_to_facts(answers_dict):
    """Converts a dictionary of question:answer_text into a list of fact strings."""
    facts = []
    for question, answer_text in answers_dict.items():
        # Standardize question key for matching
        q_map_key = question.replace('?','').strip()
        mapping = QA_TO_FACT_MAP.get(q_map_key)
        if not mapping: continue
        
        if "template" in mapping:
            facts.append(mapping["template"].format(answer_text))
        elif answer_text.lower() in {k.lower() for k in mapping.keys()}:
            for map_key, fact_string in mapping.items():
                if map_key.lower() == answer_text.lower():
                    facts.append(fact_string)
                    break
    return facts
    
def generate_test_files(entries, final_answers_lookup, args):
    """Generates the final test files based on the specified format."""
    output_data = defaultdict(list)
    print(f"Generating test files in '{args.format}' format...")

    for idx, entry in enumerate(tqdm(entries)):
        lookup_key = (entry.get("sample"), entry.get("label"))
        answers_dict = final_answers_lookup.get(lookup_key)
        if not answers_dict:
            continue

        images = entry.get("image", [])
        w, h = get_image_dimensions(images)
        image_block = "\n".join(["<image>"] * len(images))

        if args.format == 'caption_facts':
            facts = convert_answers_to_facts(answers_dict)
            if not facts: continue
            facts_block = "\n".join(facts)
            prompt_template = (
                f'{image_block}\n\nYou are given multiple images for the "{entry.get("label", "")}" phase of an accident scenario.\n\n'
                "Generate only the {} description using all the available visual cues and associated facts.\n\n"
                f'{facts_block}'
            )
            entry_data = {
                "id": f"fact_test_{idx}", "image": images, "width_list": w, "height_list": h,
                "conversations": [
                    {"from": "human", "value": prompt_template.format("pedestrian")}, {"from": "gpt", "value": ""},
                    {"from": "human", "value": prompt_template.format("vehicle")}, {"from": "gpt", "value": ""}
                ],
                "meta": {"sample": entry.get("sample"), "label": entry.get("label"), "view": "merged"}
            }
            output_data['facts'].append(entry_data)

        elif args.format == 'selective_qa':
            # Pedestrian
            vqa_ped = [{"question": q, "answer": a} for q, a in answers_dict.items() if q in ENV_QUESTIONS or q in PED_QUESTIONS or "gender" in q]
            qa_block_ped = "\n".join([f"Q: {v['question'][0].upper() + v['question'][1:]}\nA: {v['answer']}" for v in vqa_ped])
            prompt_ped = (f'{image_block}\n\nYou are given multiple images for the "{entry.get("label", "")}" phase of an accident scenario.\n\n'
                          f'Generate the pedestrian description using all the available visual cues and associated QA pairs of the scenario.\n\n{qa_block_ped}')
            output_data['ped_qa'].append({
                "id": f"ped_qa_test_{idx}", "image": images, "width_list": w, "height_list": h,
                "conversations": [{"from": "human", "value": prompt_ped}, {"from": "gpt", "value": ""}],
                "meta": {"sample": entry.get("sample"), "label": entry.get("label"), "view": "merged"}
            })
            
            # Vehicle
            vqa_veh = [{"question": q, "answer": a} for q, a in answers_dict.items() if q in ENV_QUESTIONS or q in VEH_QUESTIONS or "gender" in q or "speed" in q]
            qa_block_veh = "\n".join([f"Q: {v['question'][0].upper() + v['question'][1:]}\nA: {v['answer']}" for v in vqa_veh])
            prompt_veh = (f'{image_block}\n\nYou are given multiple images for the "{entry.get("label", "")}" phase of an accident scenario.\n\n'
                          f'Generate the vehicle description using all the available visual cues and associated QA pairs of the scenario.\n\n{qa_block_veh}')
            output_data['veh_qa'].append({
                "id": f"veh_qa_test_{idx}", "image": images, "width_list": w, "height_list": h,
                "conversations": [{"from": "human", "value": prompt_veh}, {"from": "gpt", "value": ""}],
                "meta": {"sample": entry.get("sample"), "label": entry.get("label"), "view": "merged"}
            })
            
        elif args.format == 'even_qa':
            question_keys = set(answers_dict.keys())
            dynamic_questions = {'what is the gender of pedestrian', 'what is speed of assailant vehicle'}
            if question_keys.issubset(dynamic_questions):
                continue 

            qa_block = "\n".join([f"Q: {q[0].upper() + q[1:]}\nA: {a}" for q, a in answers_dict.items()])
            prompt = (
                f'{image_block}\n\nYou are given multiple images for "{entry.get("label", "")}" phase of an accident scenario.\n\n'
                "If red and blue boxes are present in the images, the red box highlights the pedestrian and the blue box highlights the vehicle.\n\n"
                "Carefully analyze the scene using information from all available views. Focus on spatial positioning, visibility, and interactions between pedestrians and vehicles.\n\n"
                "Generate two detailed captions in the following format:\n"
                "Ped_caption: <Insert detailed pedestrian description here>\n"
                "Veh_caption: <Insert detailed vehicle description here>\n"
                "Before generating the final descriptions, consider the following annotated information:\n\n"
                f'{qa_block}'
            )
            output_data['even_qa'].append({
                "id": f"even_qa_test_{idx}", "image": images, "width_list": w, "height_list": h,
                "conversations": [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}],
                "meta": {"sample": entry.get("sample"), "label": entry.get("label"), "view": "merged"}
            })


    if args.format == 'caption_facts':
        with open(args.out_jsonl, 'w') as f:
            for item in output_data['facts']: f.write(json.dumps(item) + "\n")
        print(f"Saved {len(output_data['facts'])} entries to {args.out_jsonl}")
    elif args.format == 'selective_qa':
        with open(args.out_ped_qa_jsonl, 'w') as f:
            for item in output_data['ped_qa']: f.write(json.dumps(item) + "\n")
        print(f"Saved {len(output_data['ped_qa'])} entries to {args.out_ped_qa_jsonl}")
        with open(args.out_veh_qa_jsonl, 'w') as f:
            for item in output_data['veh_qa']: f.write(json.dumps(item) + "\n")
        print(f"Saved {len(output_data['veh_qa'])} entries to {args.out_veh_qa_jsonl}")
    elif args.format == 'even_qa':
        with open(args.out_jsonl, 'w') as f:
            for item in output_data['even_qa']: f.write(json.dumps(item) + "\n")
        print(f"Saved {len(output_data['even_qa'])} entries to {args.out_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Generate various test files from multiple references.")
    parser.add_argument('--wts-json', required=True)
    parser.add_argument('--bdd-json', required=True)
    parser.add_argument('--multiagent-vqa-jsonl', required=True)
    parser.add_argument('--test-full-ref-jsonl', required=True)
    parser.add_argument('--view-csv', help="Required for formats other than 'even_qa'.")
    parser.add_argument('--format', required=True, choices=['caption_facts', 'selective_qa', 'even_qa'])
    parser.add_argument('--out-jsonl', help="Path for single-file formats ('caption_facts', 'even_qa').")
    parser.add_argument('--out-ped-qa-jsonl', help="Path for pedestrian QA file ('selective_qa').")
    parser.add_argument('--out-veh-qa-jsonl', help="Path for vehicle QA file ('selective_qa').")
    args = parser.parse_args()


    if args.format in ['caption_facts', 'selective_qa'] and not args.view_csv:
        parser.error("--view-csv is required for 'caption_facts' and 'selective_qa' formats.")
    if args.format in ['caption_facts', 'even_qa'] and not args.out_jsonl:
        parser.error("--out-jsonl is required for 'caption_facts' and 'even_qa' formats.")
    if args.format == 'selective_qa' and (not args.out_ped_qa_jsonl or not args.out_veh_qa_jsonl):
        parser.error("--out-ped-qa-jsonl and --out-veh-qa-jsonl are required for 'selective_qa' format.")

    # --- Pipeline ---
    print("[1/5] Loading and merging input test files...")
    wts_data = merge_json_files(args.wts_json)
    bdd_data = merge_json_files(args.bdd_json)
    merged_data = wts_data + bdd_data
    
    print("\n[2/5] Building lookup tables...")
    multiagent_answers = load_multiagent_answers(args.multiagent_vqa_jsonl)
    ref_data = [json.loads(line) for line in open(args.test_full_ref_jsonl)]
    gender_speed_answers = load_gender_speed_answers(ref_data)
    vqa_options_lookup = build_vqa_options_lookup(merged_data, ref_data)
    final_answers_lookup = resolve_all_answers(multiagent_answers, gender_speed_answers, vqa_options_lookup)
    print(f"    Built final answer lookup for {len(final_answers_lookup)} samples.")

    if args.format != 'even_qa':
        print("\n[3/5] Filtering WTS overhead views...")
        valid_views = load_valid_views(args.view_csv)
        merged_data = filter_wts_overhead(merged_data, valid_views)
    
    print("\n[4/5] Merging WTS views...")
    final_entries = merge_wts_views(merged_data)
    
    print("\n[5/5] Generating final test file(s)...")
    generate_test_files(final_entries, final_answers_lookup, args)

    print("\n✨ All tasks completed successfully!")

if __name__ == "__main__":
    main()