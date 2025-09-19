import os
import json
import argparse
from collections import defaultdict
from pathlib import Path
import cv2
from tqdm import tqdm
from decord import VideoReader
from extract.caption_loader import parse_captions_from_phase
from extract.vqa_loader import load_gt_from_val_dirs
from extract.utils import (
    draw_bboxes, load_bbox_map, get_bbox_file, get_fps as get_fps_fallback
)

# -------- Phase maps --------
PHASE_MAP = {
    '0': 'prerecognition', '1': 'recognition', '2': 'judgement', '3': 'action', '4': 'avoidance',
    'prerecognition': 'prerecognition', 'recognition': 'recognition',
    'judgement': 'judgement', 'action': 'action', 'avoidance': 'avoidance'
}
# For GT lookup that may be keyed by numeric labels:
PHASE_TO_NUM = {'prerecognition': 0, 'recognition': 1, 'judgement': 2, 'action': 3, 'avoidance': 4}

# -------- Utility: evenly indices (exactly your "correct" logic) --------
def get_evenly_spaced_indices(start_frame: int, end_frame: int, num_frames: int):
    if end_frame <= start_frame or num_frames == 0:
        return []
    total = end_frame - start_frame
    step = max(total // num_frames, 1)
    return [start_frame + i * step for i in range(num_frames)]

# -------- Helper: prefer JSON fps, fallback to video probe --------
def determine_fps_from_json_or_video(caption_json: dict, video_path: str) -> float:
    fps = caption_json.get("fps", None)
    try:
        if fps is not None:
            f = float(fps)
            if f > 0:
                return f
    except Exception:
        pass
    # Fallback to cv2 probe (same as extract.utils.get_fps)
    return get_fps_fallback(video_path)

# -------- Helper: robust sample/view extraction for vqa_id wiring --------
def extract_sample_and_view(video_path: str):
    stem = Path(video_path).stem
    if stem.startswith("video"):
        return stem.split(".")[0], "vehicle"
    if "normal" in stem and "event" in stem:
        return stem, "overhead"
    if stem.endswith("vehicle_view"):
        return "_".join(stem.split("_")[:4]), "vehicle"
    return "_".join(stem.split("_")[:4]), "overhead"

# -------- Helper: JSON loader --------
def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

# -------- Conversations/VQA-ID augmentation (env priority → overhead fallback) --------
def append_conversations(frames_data, vqa_data, append_to="vqa"):
    sample_label_map = defaultdict(list)
    for entry in frames_data:
        key = (entry["sample"], entry["label"])
        sample_label_map[key].append(entry)

    conv_map = defaultdict(list)
    vqa_id_map = {}              # env-level
    overhead_fallback_map = {}   # phase-level, only if view == overhead

    for item in vqa_data:
        videos = item.get("videos", [])
        if not videos:
            continue

        sample, view = extract_sample_and_view(videos[0])

        # 1) environment-level conversations + vqa_id for all labels of same sample
        if "conversations" in item:
            for (s, label) in sample_label_map:
                if s == sample:
                    conv_map[(s, label)].extend(item["conversations"])
                    vqa_id_map[(s, label)] = videos

        # 2) phase-level fallback ONLY from overhead
        for phase in item.get("event_phase", []):
            labels = phase.get("labels", [])
            convs = phase.get("conversations", [])
            for label_raw in labels:
                label = PHASE_MAP.get(str(label_raw), f"phase_{label_raw}")
                key = (sample, label)
                if key in sample_label_map:
                    conv_map[key].extend(convs)
                    if view == "overhead" and key not in overhead_fallback_map:
                        overhead_fallback_map[key] = videos

    # 3) apply to frames
    for key in sample_label_map:
        for frame in sample_label_map[key]:
            if key in conv_map:
                frame[append_to].extend(conv_map[key])
            if key in vqa_id_map:
                frame["vqa_id"] = vqa_id_map[key]
            elif key in overhead_fallback_map:
                frame["vqa_id"] = overhead_fallback_map[key]

    return list(sample_label_map.values())

# -------- Main pipeline --------
def extract_and_draw(args):
    grouped = defaultdict(lambda: {"image": []})
    caption_root = os.path.join(args.caption_dir, args.split)
    gt_lookup = load_gt_from_val_dirs([args.gt_dir]) if args.gt_dir else {}
    caption_lookup = {}

    if args.dataset == 'WTS':
        _process_wts_evenly(caption_root, grouped, caption_lookup, args)
    elif args.dataset == 'BDD':
        _process_bdd_evenly(caption_root, grouped, caption_lookup, args)

    # Deterministic ordering across runs
    keys_sorted = sorted(grouped.keys(), key=lambda x: (x[0], x[1], x[2]))  # (sample, label, view)

    samples = []
    for idx, key in enumerate(keys_sorted):
        sample, label, view = key
        data = grouped[key]
        if not data["image"]:
            continue

        # Attach GT QAs (environment + phase) if available; try both string and numeric phase keys
        vqa = []
        env_key = (sample, "environment")
        if env_key in gt_lookup:
            vqa.extend(gt_lookup[env_key])

        # phase-level could be keyed by numeric label in GT
        label_num = PHASE_TO_NUM.get(label, None)
        if label_num is not None and (sample, label_num) in gt_lookup:
            vqa.extend(gt_lookup[(sample, label_num)])
        # also try string key just in case GT is already mapped
        if (sample, label) in gt_lookup:
            vqa.extend(gt_lookup[(sample, label)])

        caps = caption_lookup.get((sample, label), {})

        samples.append({
            "id": idx,
            "sample": sample,
            "label": label,
            "view": view,
            "image": data["image"],
            "num_images": len(data["image"]),
            "vqa": vqa,
            "caption_pedestrian": caps.get("caption_pedestrian", ""),
            "caption_vehicle": caps.get("caption_vehicle", "")
        })

    # Test-only: augment with provided VQA JSON (conversations + vqa_id)
    if args.split == "test" and args.vqa_json:
        vqa_data = load_json(args.vqa_json)
        samples_nested = append_conversations(samples, vqa_data, append_to="vqa")
        samples = [item for sublist in samples_nested for item in sublist]

    # Save
    out_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_{args.split}_frames.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"[DONE] Saved {len(samples)} samples to {out_path}")

def _process_wts_evenly(caption_root, grouped, caption_lookup, args):
    samples = os.listdir(caption_root)
    if "normal_trimmed" in samples:
        samples.remove("normal_trimmed")

    # Regular WTS samples
    for sample_name in tqdm(samples, desc="WTS Samples"):
        _process_wts_sample_evenly(
            sample_name,
            os.path.join(caption_root, sample_name),
            grouped,
            caption_lookup,
            args,
            sample_parent=""
        )

    # normal_trimmed subtree
    nt_path = os.path.join(caption_root, "normal_trimmed")
    if os.path.isdir(nt_path):
        for sub_sample in tqdm(os.listdir(nt_path), desc="normal_trimmed"):
            _process_wts_sample_evenly(
                sub_sample,
                os.path.join(nt_path, sub_sample),
                grouped,
                caption_lookup,
                args,
                sample_parent="normal_trimmed"
            )

def _process_wts_sample_evenly(sample_name, sample_path, grouped, caption_lookup, args, sample_parent=""):
    for view in ["overhead_view", "vehicle_view"]:
        caption_dir = os.path.join(sample_path, view)
        if not os.path.exists(caption_dir):
            continue

        for fname in os.listdir(caption_dir):
            if not fname.endswith("_caption.json"):
                continue

            cpath = os.path.join(caption_dir, fname)
            try:
                data = json.load(open(cpath))
            except Exception:
                continue

            view_folder = "overhead" if view == "overhead_view" else "vehicle"
            video_list = data.get("overhead_videos", []) if view_folder == "overhead" else [f"{sample_name}_vehicle_view.mp4"]

            # captions per phase
            caption_map = parse_captions_from_phase(data)

            for video_name in video_list:
                video_path = os.path.join(args.video_dir, args.split, sample_parent, sample_name, f"{view_folder}_view", video_name)
                if not os.path.exists(video_path):
                    continue

                try:
                    vr = VideoReader(video_path)
                except Exception:
                    continue

                fps = determine_fps_from_json_or_video(data, video_path)

                # bbox maps (WTS)
                ped_bbox_file = get_bbox_file(args.bbox_dir, 'WTS', args.split, view, sample_parent, sample_name, video_name.replace(".mp4", ""), "pedestrian")
                veh_bbox_file = get_bbox_file(args.bbox_dir, 'WTS', args.split, view, sample_parent, sample_name, video_name.replace(".mp4", ""), "vehicle")
                ped_bboxes = load_bbox_map(ped_bbox_file)
                veh_bboxes = load_bbox_map(veh_bbox_file)

                for phase in data.get("event_phase", []):
                    label = PHASE_MAP.get(str(phase["labels"][0]), f"phase_{phase['labels'][0]}")

                    # Reproducible EVENLY: int(start*fps), int(end*fps)
                    start_frame = int(float(phase["start_time"]) * fps)
                    end_frame   = int(float(phase["end_time"])   * fps)
                    frame_ids = get_evenly_spaced_indices(start_frame, end_frame, args.num_frames)

                    save_path = os.path.join(args.output_dir, args.split, sample_name, view_folder, video_name.replace(".mp4", ""), label)

                    for fid in frame_ids:
                        frame_path = os.path.join(save_path, f"frame_{fid}.jpg")
                        if not os.path.exists(frame_path):
                            try:
                                frame = vr[fid].asnumpy()
                                frame = draw_bboxes(frame, ped_bboxes.get(fid, []), (255, 0, 0), "Pedestrian")
                                frame = draw_bboxes(frame, veh_bboxes.get(fid, []), (0, 0, 255), "Vehicle")
                                if args.save:
                                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            except Exception:
                                continue
                        grouped[(sample_name, label, view_folder)]["image"].append(frame_path)

                    # cache captions for (sample,label)
                    caption_lookup[(sample_name, label)] = caption_map.get(label, {})

def _process_bdd_evenly(caption_root, grouped, caption_lookup, args):
    for root, _, files in os.walk(caption_root):
        caption_files = [f for f in files if f.endswith("_caption.json")]
        for fname in tqdm(caption_files, desc="BDD Samples"):
            caption_path = os.path.join(root, fname)
            try:
                data = json.load(open(caption_path))
            except Exception:
                continue

            sample_name = data['video_name'].replace(".mp4", "")
            video_path = os.path.join(args.video_dir, args.split, data['video_name'])
            if not os.path.exists(video_path):
                continue

            try:
                vr = VideoReader(video_path)
            except Exception:
                continue

            fps = determine_fps_from_json_or_video(data, video_path)
            caption_map = parse_captions_from_phase(data)

            # BDD bboxes: one merged file (ped only), path pattern per your "correct" script
            bbox_file = os.path.join(args.bbox_dir, args.split, f"{sample_name}_bbox.json")
            ped_bboxes = load_bbox_map(bbox_file)

            for phase in data.get("event_phase", []):
                label = PHASE_MAP.get(str(phase["labels"][0]), f"phase_{phase['labels'][0]}")
                start_frame = int(float(phase["start_time"]) * fps)
                end_frame   = int(float(phase["end_time"])   * fps)
                frame_ids = get_evenly_spaced_indices(start_frame, end_frame, args.num_frames)

                save_path = os.path.join(args.output_dir, args.split, sample_name, "vehicle", label)

                for fid in frame_ids:
                    frame_path = os.path.join(save_path, f"frame_{fid}.jpg")
                    if not os.path.exists(frame_path):
                        try:
                            frame = vr[fid].asnumpy()
                            frame = draw_bboxes(frame, ped_bboxes.get(fid, []), (255, 0, 0), "Pedestrian")
                            if args.save:
                                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        except Exception:
                            continue
                    grouped[(sample_name, label, "vehicle")]["image"].append(frame_path)

                caption_lookup[(sample_name, label)] = caption_map.get(label, {})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['WTS', 'BDD'], required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--video-dir', type=str, required=True)
    parser.add_argument('--caption-dir', type=str, required=True)
    parser.add_argument('--bbox-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--gt-dir', type=str, default=None, help="GT QA root directory")
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--num-frames', type=int, default=5,
                        help="Number of frames to sample for 'evenly' strategy")
    parser.add_argument('--vqa-json', type=str, default=None,
                        help="Optional VQA JSON file (only for test split)")

    args = parser.parse_args()
    extract_and_draw(args)