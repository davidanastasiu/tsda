import argparse
import json
from pathlib import Path
from tqdm import tqdm
import re

# --- Question Categorization Sets (for 'selective_QA' format) ---
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

    "what is the pedestrian's line of sight?": {"template": "Pedestrian's line of sight was focused on {}."},
    "what is the pedestrian's visual status?": {"template": "The pedestrian is {}."},

    "what is the pedestrian's awareness regarding vehicle?": {
        "Notices the vehicle": "The pedestrian appears to be aware of the vehicle.",
        "Unaware of the vehicle": "The pedestrian does not appear to be aware of the vehicle.",
        "Cannot be determined": "The pedestrian's awareness regarding the vehicle cannot be determined."
    },

    "what is the pedestrian's action?": {"template": "The pedestrian is {}."},

    # Vehicle-side
    "what is the position of the vehicle relative to the pedestrian?": {"template": "The vehicle is positioned {} relative to the pedestrian."},
    "what is relative distance of vehicle from pedestrian?": {"template": "The vehicle is at a {} distance from the pedestrian."},
    "what is vehicle's field of view?": {
        "Pedestrian is visible": "The pedestrian is visible in the vehicle's field of view.",
        "Pedestrian is not visible": "The pedestrian is not visible in the vehicle's field of view."
    },
    "what is the action taken by vehicle?": {"template": "The vehicle is {}."},

    # Dynamic questions filled from full-ref
    "what is the gender of the pedestrian?": {"template": "The pedestrian is {}."},
    "what is the speed of the assailant vehicle?": {"template": "The assailant vehicle is moving at {}."},

    "what is the pedestrian's direction of travel?": {"template": "The pedestrian's direction of travel is {}."},
    "what is pedestrian's speed?": {"template": "The pedestrian is moving {}."},
    "what is the fine-grained action taken by the pedestrian?": {"template": "The pedestrian is {}."},
    "is the pedestrian wearing glasses?": {"Yes": "The pedestrian is wearing glasses.", "No": "The pedestrian is not wearing glasses."},

    "what is the setting of this scenario?": {"template": "The scenario is set in a {} environment."},
    "what is the formation of the road?": {"template": "The road is {}."},
    "is there guardrail around the road?": {"Yes": "There is a guardrail around the road.", "No": "There is no guardrail around the road."},
    "is the pedestrian holding a waling cane?": {"Yes": "The pedestrian is holding a walking cane.", "No": "The pedestrian is not holding a walking cane."}
}


def load_best_answers(best_answers_path):
    """Loads the best answers file into a lookup dictionary using (sample, label) as the key."""
    answers_map = {}
    with open(best_answers_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            meta = item.get("meta", {})
            sample_id = meta.get("sample")
            label = meta.get("label")
            if sample_id and label:
                key = (sample_id, label)
                answers_map[key] = {p.get("question", "").lower().strip(): p.get("answer", "") for p in item.get("predictions", [])}
    return answers_map


def format_vqa_to_qa_block(vqa_list):
    """Formats a list of {'question': q, 'answer': a} dicts into a Q:/A: string block."""
    return "\n".join([f"Q: {v['question'][0].upper() + v['question'][1:]}\nA: {v['answer']}" for v in vqa_list if v.get("question") and v.get("answer")])


def convert_answers_to_facts(answers_dict):
    """Converts a dictionary of answers into a list of fact strings."""
    facts = []
    for question, answer in answers_dict.items():
        if not answer: continue
        mapping = QA_TO_FACT_MAP.get(question)
        if not mapping: continue
        if "template" in mapping:
            facts.append(mapping["template"].format(answer))
        elif answer.lower() in {key.lower() for key in mapping.keys()}:
            for map_key, fact_string in mapping.items():
                if map_key.lower() == answer.lower():
                    facts.append(fact_string)
                    break
    return facts


def get_original_captions(entry):
    """Extracts original ped/veh captions from the gpt response field."""
    ped_caption, veh_caption = "", ""
    human_prompt = entry['conversations'][0]['value']
    gpt_response = entry['conversations'][1]['value']

    if "Ped_caption:" in gpt_response: # even_qa format
        try:
            ped_caption = gpt_response.split("Ped_caption:")[1].split("Veh_caption:")[0].strip()
            veh_caption = gpt_response.split("Veh_caption:")[1].strip()
        except IndexError:
            pass
    elif "Generate the" in human_prompt and "associated QA pairs" in human_prompt: # selective_qa format
        if "pedestrian description" in human_prompt:
            ped_caption = gpt_response
        else:
            veh_caption = gpt_response
    elif "Generate only the" in human_prompt and len(entry['conversations']) > 2: # caption_facts format
        ped_caption = gpt_response
        veh_caption = entry['conversations'][3]['value']
            
    return ped_caption, veh_caption


def update_val_file(input_path, best_answers_map, output_path, file_format):
    """Conditionally updates a val file, preserving row count and dynamic QAs."""
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in tqdm(f_in, desc=f"Processing {Path(input_path).name}"):
            entry = json.loads(line)
            meta = entry.get("meta", {})
            lookup_key = (meta.get("sample"), meta.get("label"))
            
            # --- Step 1: Extract original captions BEFORE any modifications ---
            ped_caption, veh_caption = get_original_captions(entry)
            
            best_answers_for_entry = best_answers_map.get(lookup_key)
            
            if best_answers_for_entry:
                if file_format == 'even_qa':
                    vqa_list = [{"question": q, "answer": a} for q, a in best_answers_for_entry.items()]
                    # Now add gender/speed using the extracted captions
                    gender_ans = extract_answer_from_caption("gender", ped_caption, veh_caption)
                    speed_ans = extract_answer_from_caption("speed", ped_caption, veh_caption)
                    vqa_list.append({"question": "What is the gender of Pedestrian?", "answer": gender_ans})
                    vqa_list.append({"question": "What is speed of Assailant Vehicle?", "answer": speed_ans})

                    qa_block = format_vqa_to_qa_block(vqa_list)
                    human_prompt_parts = entry['conversations'][0]['value'].split("Before generating the final descriptions,")
                    entry['conversations'][0]['value'] = f"{human_prompt_parts[0]}Before generating the final descriptions, consider the following annotated information:\n\n{qa_block}"

                elif file_format == 'selective_qa':
                    is_ped_file = "Generate the pedestrian description" in entry['conversations'][0]['value']
                    vqa_list = []
                    
                    if is_ped_file:
                        vqa_list = [{"question": q, "answer": a} for q, a in best_answers_for_entry.items() if q in ENV_QUESTIONS or q in PED_QUESTIONS]
                        gender_ans = extract_answer_from_caption("gender", ped_caption, "")
                        vqa_list.append({"question": "What is the gender of Pedestrian?", "answer": gender_ans})
                    else: # Vehicle file
                        vqa_list = [{"question": q, "answer": a} for q, a in best_answers_for_entry.items() if q in VEH_QUESTIONS or q in ENV_QUESTIONS]
                        gender_ans = extract_answer_from_caption("gender", ped_caption, veh_caption)
                        speed_ans = extract_answer_from_caption("speed", ped_caption, veh_caption)
                        vqa_list.append({"question": "What is the gender of Pedestrian?", "answer": gender_ans})
                        vqa_list.append({"question": "What is speed of Assailant Vehicle?", "answer": speed_ans})

                    qa_block = format_vqa_to_qa_block(vqa_list)
                    desc_type = "pedestrian" if is_ped_file else "vehicle"
                    label = meta.get("label", "default")
                    image_block = "\n".join(["<image>"] * len(entry["image"]))
                    
                    new_prompt = (
                        f'{image_block}\n\nYou are given multiple images for the "{label}" phase of an accident scenario.\n\n'
                        f"Generate the {desc_type} description using all the available visual cues and associated QA pairs of the scenario.\n\n"
                        f'{qa_block}'
                    )
                    entry['conversations'][0]['value'] = new_prompt
                
                elif file_format == 'caption_facts':
                    answers_with_dynamic = dict(best_answers_for_entry)
                    # Use extracted captions to get gender/speed
                    answers_with_dynamic["what is the gender of the pedestrian?"] = extract_answer_from_caption("gender", ped_caption, "")
                    answers_with_dynamic["what is the speed of the assailant vehicle?"] = extract_answer_from_caption("speed", "", veh_caption)

                    facts_block = "\n".join(convert_answers_to_facts(answers_with_dynamic))
                    human_prompt_parts = entry['conversations'][0]['value'].split("\n\nGenerate only the")
                    base_structure = human_prompt_parts[0]
                    
                    entry['conversations'][0]['value'] = f"{base_structure}\n\nGenerate only the pedestrian description using all the available visual cues and associated facts.\n\n{facts_block}"
                    entry['conversations'][2]['value'] = f"{base_structure}\n\nGenerate only the vehicle description using all the available visual cues and associated facts.\n\n{facts_block}"
            
            for i, conv in enumerate(entry['conversations']):
                if conv['from'] == 'gpt':
                    entry['conversations'][i]['value'] = ""

            f_out.write(json.dumps(entry) + "\n")

    print(f"✅ Successfully created inference file at {output_path}")

def extract_answer_from_caption(question, ped_caption, veh_caption):
    """Helper to get gender/speed from original captions."""
    question = question.lower()
    caption_for_gender = ped_caption if ped_caption else veh_caption
    
    if "gender" in question:
        text = caption_for_gender.lower()
        if any(w in text for w in ["female", "woman", "women"]): return "Female"
        return "Male"
    if "speed" in question:
        # Use veh_caption specifically for speed
        if veh_caption:
            match = re.search(r"(\d+)\s*km/?h", veh_caption)
            if match: return f"{match.group(1)} km/h"
        return "0 km/h"
    return ""


def main():
    parser = argparse.ArgumentParser(description="Update validation files with best answers for inference, preserving row count.")
    parser.add_argument('--input-val-jsonl', required=True, help="Path to the original validation JSONL file to update.")
    parser.add_argument('--best-answers-jsonl', required=True, help="Path to the best answers JSONL file.")
    parser.add_argument('--output-jsonl', required=True, help="Path for the updated, inference-ready output file.")
    parser.add_argument('--format', required=True, choices=['even_qa', 'selective_qa', 'caption_facts'], 
                        help="The format of the input/output file.")
    args = parser.parse_args()

    print(f"[1/2] Loading best answers from {args.best_answers_jsonl}...")
    best_answers_map = load_best_answers(args.best_answers_jsonl)
    print(f"Loaded answers for {len(best_answers_map)} unique (sample, label) pairs.")

    print(f"\n[2/2] Processing {args.input_val_jsonl} with format '{args.format}'...")
    update_val_file(args.input_val_jsonl, best_answers_map, args.output_jsonl, args.format)
    
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()