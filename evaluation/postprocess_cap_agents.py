#!/usr/bin/env python3
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )
    logging.getLogger().addHandler(logging.StreamHandler())  

def find_rank_files(model_dir: Path) -> List[Path]:
    candidates = list(model_dir.glob("**/*.jsonl"))
    ranked = []
    for p in candidates:
        m = re.search(r"rank[_\-]?(\d+)", p.stem, re.IGNORECASE)
        if m:
            ranked.append((int(m.group(1)), p))
    ranked.sort(key=lambda t: t[0])
    return [p for _, p in ranked]

def merge_rank_jsonls(rank_files: List[Path], merged_path: Path) -> int:
    """
    Concatenate rank*.jsonl into one merged file in rank order.
    Adds/overwrites a top-level key 'rank_source' on each line for traceability.
    Returns number of lines written.
    """
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with merged_path.open("w") as out_f:
        for rf in rank_files:
            m = re.search(r"rank[_\-]?(\d+)", rf.stem, re.IGNORECASE)
            rank_id = int(m.group(1)) if m else -1
            with rf.open("r") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {rf}: {line[:120]}...")
                        continue
                    obj["rank_source"] = rank_id
                    out_f.write(json.dumps(obj) + "\n")
                    count += 1
    logging.info(f"Merged {count} lines from {len(rank_files)} rank files into {merged_path}")
    return count


LABEL_MAP = {
    "prerecognition": "0",
    "recognition": "1",
    "judgement": "2",
    "action": "3",
    "avoidance": "4"
}


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with path.open("r") as f:
            return json.load(f)
    return {}

def _save_json(path: Path, data: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)

def _merge_entry(target: Dict[str, Any], sample: str, label_id: str,
                 ped_caption: Optional[str], veh_caption: Optional[str]) -> None:
   
    if sample not in target:
        target[sample] = []

    # Try to find an existing entry with the same label_id
    slot = None
    for item in target[sample]:
        if isinstance(item, dict) and item.get("labels") == [label_id]:
            slot = item
            break

    if slot is None:
        slot = {
            "labels": [label_id],
            "caption_pedestrian": "",
            "caption_vehicle": ""
        }
        target[sample].append(slot)

    if ped_caption and not slot.get("caption_pedestrian"):
        slot["caption_pedestrian"] = ped_caption
    if veh_caption and not slot.get("caption_vehicle"):
        slot["caption_vehicle"] = veh_caption

# -----------------------------
# Mode 1: New Captions (your provided logic)
# -----------------------------

def _extract_ped_veh_from_mid_facts(conversations) -> Tuple[Optional[str], Optional[str]]:
    """
    mid_facts format: two separate HUMAN prompts:
      - 'Generate only the pedestrian description ...'
      - 'Generate only the vehicle description ...'
    Each followed by its own GPT reply. Return (ped_caption, veh_caption).
    """
    if not isinstance(conversations, list):
        return None, None

    expect = None   # 'ped' or 'veh'
    ped_cap = None
    veh_cap = None

    for turn in conversations:
        role = (turn.get("from") or "").lower()
        val = (turn.get("value") or "").strip()
        if not val:
            continue

        if role == "human":
            low = val.lower()
            if "generate only the pedestrian description" in low:
                expect = "ped"
            elif "generate only the vehicle description" in low:
                expect = "veh"
        elif role == "gpt":
            if expect == "ped" and ped_cap is None:
                ped_cap = val
                expect = None
            elif expect == "veh" and veh_cap is None:
                veh_cap = val
                expect = None

    return ped_cap, veh_cap

def postprocess_mid_facts_caption(infer_jsonl_path: Path) -> Dict[str, Any]:
    
    output = defaultdict(list)
    sample_label_counter = defaultdict(set)

    total_lines = 0
    missing_both = 0
    missing_ped = 0
    missing_veh = 0

    with infer_jsonl_path.open("r") as f:
        for line in f:
            total_lines += 1
            item = json.loads(line.strip())

            meta = item.get("meta", {})
            sample = meta.get("sample") or meta.get('scene') or meta.get('id')
            if sample is None:
                logging.warning(f"Line {total_lines}: missing meta.sample; skipping.")
                continue

            label_name = meta.get("label")
            if label_name not in LABEL_MAP:
                logging.warning(f"Sample {sample}: unknown/missing label '{label_name}'; skipping.")
                continue
            label_id = LABEL_MAP[label_name]

            conversations = item.get("conversations", [])
            ped_cap, veh_cap = _extract_ped_veh_from_mid_facts(conversations)

            if not ped_cap and not veh_cap:
                missing_both += 1
                logging.warning(f"Sample {sample}: could not extract ped/veh captions from mid_facts.")
                continue

            if not ped_cap:
                missing_ped += 1
                logging.warning(f"Sample {sample}: missing Ped_caption (mid_facts).")
            if not veh_cap:
                missing_veh += 1
                logging.warning(f"Sample {sample}: missing Veh_caption (mid_facts).")

            sample_label_counter[sample].add(label_id)
            output[sample].append({
                "labels": [label_id],
                "caption_pedestrian": ped_cap or "",
                "caption_vehicle": veh_cap or ""
            })

    logging.info("\n[Mode mid_facts] Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Total unique samples: {len(output)}")
    logging.info(f"Completely missing (both): {missing_both}")
    logging.info(f"Missing Ped_caption only: {missing_ped}")
    logging.info(f"Missing Veh_caption only: {missing_veh}")
    for sample, labels in sample_label_counter.items():
        logging.info(f"Sample: {sample}, Labels found: {sorted(labels)}")

    return output


def postprocess_new_captions(infer_jsonl_path: Path) -> Dict[str, Any]:
    output = defaultdict(list)
    sample_label_counter = defaultdict(set)

    total_lines = 0
    missing_ped = 0
    missing_veh = 0

    with infer_jsonl_path.open('r') as f:
        for line in f:
            total_lines += 1
            item = json.loads(line.strip())

            meta = item.get('meta', {})
            sample = meta.get('sample') or meta.get('scene') or meta.get('id')
            if sample is None:
                logging.warning(f"Missing 'meta.sample' in line {total_lines}, skipping")
                continue

            label_name = meta.get('label')
            if label_name not in LABEL_MAP:
                logging.warning(f"Unknown or missing label '{label_name}' in sample {sample}")
                continue
            label_id = LABEL_MAP[label_name]

            caption_pedestrian = ""
            caption_vehicle = ""

            for pred in item.get("predictions", []):
                q = (pred.get("question") or "").lower().strip()
                a = (pred.get("answer") or "").strip()
                if q == "ped_caption":
                    caption_pedestrian = a
                elif q == "veh_caption":
                    caption_vehicle = a

            if not caption_pedestrian:
                missing_ped += 1
                logging.warning(f"Missing Ped_caption in sample {sample}, label {label_name}")
            if not caption_vehicle:
                missing_veh += 1
                logging.warning(f"Missing Veh_caption in sample {sample}, label {label_name}")

            sample_label_counter[sample].add(label_id)
            output[sample].append({
                "labels": [label_id],
                "caption_pedestrian": caption_pedestrian,
                "caption_vehicle": caption_vehicle
            })

    logging.info("\n[Mode new_captions] Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Total unique samples: {len(output)}")
    logging.info(f"Missing Ped_caption entries: {missing_ped}")
    logging.info(f"Missing Veh_caption entries: {missing_veh}")
    for sample, labels in sample_label_counter.items():
        logging.info(f"Sample: {sample}, Labels found: {sorted(labels)}")

    return output

# -----------------------------
# Mode 2: even_QA_caption (conversations → two captions)
# -----------------------------

_PED_VEH_REGEX = re.compile(
    r"(?is)ped_caption\s*:\s*(.*?)\s*veh_caption\s*:\s*(.*)"
)

def _extract_captions_from_gpt(conversations) -> Tuple[Optional[str], Optional[str]]:
   
    if not isinstance(conversations, list):
        return None, None

    last_gpt_text = None
    for turn in conversations:
        if isinstance(turn, dict) and (turn.get("from") or "").lower() == "gpt":
            last_gpt_text = turn.get("value") or last_gpt_text

    if not last_gpt_text:
        return None, None

    m = _PED_VEH_REGEX.search(last_gpt_text)
    if m:
        ped = m.group(1).strip()
        veh = m.group(2).strip()
        return ped, veh

    
    m2 = re.search(r"(?is)veh_caption\s*:\s*(.*?)\s*ped_caption\s*:\s*(.*)", last_gpt_text)
    if m2:
        veh = m2.group(1).strip()
        ped = m2.group(2).strip()
        return ped, veh

    return None, None

def postprocess_even_qa_caption(infer_jsonl_path: Path) -> Dict[str, Any]:
    output = defaultdict(list)
    sample_label_counter = defaultdict(set)

    total_lines = 0
    missing_ped = 0
    missing_veh = 0
    missing_both = 0

    with infer_jsonl_path.open("r") as f:
        for line in f:
            total_lines += 1
            item = json.loads(line.strip())

            meta = item.get("meta", {})
            sample = meta.get("sample") or meta.get('scene') or meta.get('id')
            if sample is None:
                logging.warning(f"Line {total_lines}: missing meta.sample; skipping.")
                continue

            label_name = meta.get("label")
            if label_name not in LABEL_MAP:
                logging.warning(f"Sample {sample}: unknown/missing label '{label_name}'; skipping.")
                continue
            label_id = LABEL_MAP[label_name]

            conversations = item.get("conversations", [])
            ped_cap, veh_cap = _extract_captions_from_gpt(conversations)

            if not ped_cap and not veh_cap:
                missing_both += 1
                logging.warning(f"Sample {sample}: could not parse both captions from GPT message.")
                continue

            if not ped_cap:
                missing_ped += 1
                logging.warning(f"Sample {sample}: missing Ped_caption.")
            if not veh_cap:
                missing_veh += 1
                logging.warning(f"Sample {sample}: missing Veh_caption.")

            sample_label_counter[sample].add(label_id)
            output[sample].append({
                "labels": [label_id],
                "caption_pedestrian": ped_cap or "",
                "caption_vehicle": veh_cap or ""
            })

    logging.info("\n[Mode even_qa_caption] Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Total unique samples: {len(output)}")
    logging.info(f"Completely missing (both): {missing_both}")
    logging.info(f"Missing Ped_caption only: {missing_ped}")
    logging.info(f"Missing Veh_caption only: {missing_veh}")
    for sample, labels in sample_label_counter.items():
        logging.info(f"Sample: {sample}, Labels found: {sorted(labels)}")

    return output

# -----------------------------
# Mode 3a: ped_qa_caption (last GPT turn → pedestrian caption only)
# -----------------------------

def _extract_last_gpt_text(conversations) -> Optional[str]:
    if not isinstance(conversations, list):
        return None
    last = None
    for turn in conversations:
        if isinstance(turn, dict) and (turn.get("from") or "").lower() == "gpt":
            last = turn.get("value") or last
    return last

def postprocess_ped_qa_caption(infer_jsonl_path: Path) -> Dict[str, Any]:
    output = defaultdict(list)
    total_lines = 0
    missing = 0

    with infer_jsonl_path.open("r") as f:
        for line in f:
            total_lines += 1
            item = json.loads(line.strip())

            meta = item.get("meta", {})
            sample = meta.get("sample") or meta.get('scene') or meta.get('id')
            if sample is None:
                logging.warning(f"Line {total_lines}: missing meta.sample; skipping.")
                continue

            label_name = meta.get("label")
            if label_name not in LABEL_MAP:
                logging.warning(f"Sample {sample}: unknown/missing label '{label_name}'; skipping.")
                continue
            label_id = LABEL_MAP[label_name]

            gpt_text = _extract_last_gpt_text(item.get("conversations"))
            if not gpt_text:
                missing += 1
                logging.warning(f"Sample {sample}: no GPT text; skipping.")
                continue

            # Pedestrian caption only
            output[sample].append({
                "labels": [label_id],
                "caption_pedestrian": gpt_text.strip(),
                "caption_vehicle": ""
            })

    logging.info("\n[Mode ped_qa_caption] Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Ped-only captions written: {sum(len(v) for v in output.values())}")
    logging.info(f"Missing GPT text: {missing}")
    return output

# -----------------------------
# Mode 3b: veh_qa_caption (last GPT turn → vehicle caption only)
# -----------------------------

def postprocess_veh_qa_caption(infer_jsonl_path: Path) -> Dict[str, Any]:
    output = defaultdict(list)
    total_lines = 0
    missing = 0

    with infer_jsonl_path.open("r") as f:
        for line in f:
            total_lines += 1
            item = json.loads(line.strip())

            meta = item.get("meta", {})
            sample = meta.get("sample") or meta.get('scene') or meta.get('id')
            if sample is None:
                logging.warning(f"Line {total_lines}: missing meta.sample; skipping.")
                continue

            label_name = meta.get("label")
            if label_name not in LABEL_MAP:
                logging.warning(f"Sample {sample}: unknown/missing label '{label_name}'; skipping.")
                continue
            label_id = LABEL_MAP[label_name]

            gpt_text = _extract_last_gpt_text(item.get("conversations"))
            if not gpt_text:
                missing += 1
                logging.warning(f"Sample {sample}: no GPT text; skipping.")
                continue

            # Vehicle caption only
            output[sample].append({
                "labels": [label_id],
                "caption_pedestrian": "",
                "caption_vehicle": gpt_text.strip()
            })

    logging.info("\n[Mode veh_qa_caption] Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Veh-only captions written: {sum(len(v) for v in output.values())}")
    logging.info(f"Missing GPT text: {missing}")
    return output


def process_model_dir(model_dir: Path, out_root: Path, mode: str,
                      final_output_path: Optional[Path]) -> Dict[str, Any]:
    model_name = model_dir.name
    merged_dir = out_root / model_name / "merged"
    outputs_dir = out_root / model_name / "outputs"
    logs_dir = out_root / model_name / "logs"

    rank_files = find_rank_files(model_dir)
    if not rank_files:
        print(f"[WARN] No rank*.jsonl files found under {model_dir}. Skipping.")
        return {}

    merged_path = merged_dir / "merged.jsonl"
    merge_rank_jsonls(rank_files, merged_path)

    log_path = logs_dir / f"{mode}.log"
    setup_logging(log_path)  # ensure logging for this model/mode
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Run mode-specific postprocessor to get a per-model dict
    if mode == "new_captions":
        local_output = postprocess_new_captions(merged_path)
        default_model_output = outputs_dir / "new_captions_output.json"
    elif mode == "even_qa_caption":
        local_output = postprocess_even_qa_caption(merged_path)
        default_model_output = outputs_dir / "even_qa_caption_output.json"
    elif mode == "ped_qa_caption":
        local_output = postprocess_ped_qa_caption(merged_path)
        default_model_output = outputs_dir / "ped_qa_caption_output.json"
    elif mode == "veh_qa_caption":
        local_output = postprocess_veh_qa_caption(merged_path)
        default_model_output = outputs_dir / "veh_qa_caption_output.json"
    elif mode == "mid_facts":
        local_output = postprocess_mid_facts_caption(merged_path)
        default_model_output = outputs_dir / "mid_facts_caption_output.json"

    else:
        raise ValueError(f"Unknown mode: {mode}")

   
    _save_json(default_model_output, local_output)
    print(f"Per-model output saved: {default_model_output}")

   
    if final_output_path:
        final_data = _load_json(final_output_path)
        for sample, items in local_output.items():
            for item in items:
                label_list = item.get("labels") or []
                label_id = label_list[0] if label_list else None
                if label_id is None:
                    continue
                _merge_entry(
                    final_data,
                    sample,
                    label_id,
                    item.get("caption_pedestrian"),
                    item.get("caption_vehicle")
                )
        _save_json(final_output_path, final_data)
        print(f"Appended into final output: {final_output_path}")

    return local_output

def main():
    parser = argparse.ArgumentParser(description="Merge rank predictions and postprocess for multiple model directories.")
    parser.add_argument(
        "--model_dirs",
        nargs="+",
        required=True,
        help="Paths to model directories (each containing rank*.jsonl files)"
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Root output directory (merged files, per-model outputs, logs will be nested per model)"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["new_captions", "even_qa_caption", "ped_qa_caption", "veh_qa_caption", "mid_facts"],
        help="Which postprocessor to run after merging"
    )
    parser.add_argument(
        "--final_output",
        help="(Optional) Path to a SINGLE final JSON collecting/merging captions from multiple runs/modes"
    )
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    final_output_path = Path(args.final_output) if args.final_output else None

    for md in args.model_dirs:
        process_model_dir(Path(md), out_root, args.mode, final_output_path)

if __name__ == "__main__":
    main()
