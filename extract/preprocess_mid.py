import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm
from decord import VideoReader
import cv2
import math
from pathlib import Path

from extract.utils import get_centered_frames, get_fps, get_evenly_spaced_indices, draw_bboxes, load_bbox_map, get_bbox_file
from extract.vqa_loader import load_gt_from_val_dirs
from extract.caption_loader import parse_captions_from_phase

PHASE_MAP = {
    '0': 'prerecognition', '1': 'recognition', '2': 'judgement', '3': 'action', '4': 'avoidance',
    'prerecognition': 'prerecognition', 'recognition': 'recognition', 'judgement': 'judgement',
    'action': 'action', 'avoidance': 'avoidance'
}

def extract_sample_and_view(video_path: str):
    stem = Path(video_path).stem
    if stem.startswith("video"):
        return stem.split(".")[0], "vehicle"
    if "normal" in stem and "event" in stem:
        return stem, "overhead"
    if stem.endswith("vehicle_view"):
        return "_".join(stem.split("_")[:4]), "vehicle"
    return "_".join(stem.split("_")[:4]), "overhead"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def append_conversations(frames_data, vqa_data, append_to="vqa"):
    from collections import defaultdict

    sample_label_map = defaultdict(list)
    for entry in frames_data:
        key = (entry["sample"], entry["label"])
        sample_label_map[key].append(entry)

    conv_map = defaultdict(list)
    vqa_id_map = {}                 # from environment-level
    overhead_fallback_map = {}     # from overhead view only

    for item in vqa_data:
        videos = item.get("videos", [])
        if not videos:
            continue

        sample, view = extract_sample_and_view(videos[0])

        
        if "conversations" in item:
            for (s, label) in sample_label_map:
                if s == sample:
                    conv_map[(s, label)].extend(item["conversations"])
                    vqa_id_map[(s, label)] = videos

        
        for phase in item.get("event_phase", []):
            labels = phase.get("labels", [])
            convs = phase.get("conversations", [])
            for label in labels:
                key = (sample, PHASE_MAP.get(str(label), f"phase_{label}"))
                if key in sample_label_map:
                    conv_map[key].extend(convs)
                    if view == "overhead" and key not in overhead_fallback_map:
                        overhead_fallback_map[key] = videos  

   
    for key in sample_label_map:
        for frame in sample_label_map[key]:
            if key in conv_map:
                frame[append_to].extend(conv_map[key])

            if key in vqa_id_map:
                frame["vqa_id"] = vqa_id_map[key]
            elif key in overhead_fallback_map:
                frame["vqa_id"] = overhead_fallback_map[key]

    return list(sample_label_map.values())


def extract_and_draw(args):
    grouped = defaultdict(lambda: {"image": []})
    caption_root = os.path.join(args.caption_dir, args.split)
    gt_lookup = load_gt_from_val_dirs([args.gt_dir]) if args.gt_dir else {}

    caption_lookup = {}

    # Step 1: Extract all frame paths + captions per dataset
    if args.dataset == 'WTS':
        _process_wts(caption_root, grouped, caption_lookup, args, gt_lookup)
    elif args.dataset == 'BDD':
        _process_bdd(caption_root, grouped, caption_lookup, args, gt_lookup)

    # Step 2: Construct initial sample entries
    samples = []
    for idx, ((sample, label, view), data) in enumerate(grouped.items()):
        if data["image"]:
            vqa_key = (sample, label)
            env_key = (sample, "environment")
            vqa = []
            if env_key in gt_lookup:
                vqa.extend(gt_lookup[env_key])
            if vqa_key in gt_lookup:
                vqa.extend(gt_lookup[vqa_key])

            captions = caption_lookup.get((sample, label), {})

            sample_entry = {
                "id": idx,
                "sample": sample,
                "label": label,
                "view": view,
                "image": data["image"],
                "num_images": len(data["image"]),
                "vqa": vqa,
                "caption_pedestrian": captions.get("caption_pedestrian", ""),
                "caption_vehicle": captions.get("caption_vehicle", "")
            }
            samples.append(sample_entry)

    # Step 3: For test set only – load extra vqa/conversations from provided file
    if args.split == "test" and args.vqa_json:
        vqa_data = load_json(args.vqa_json)
        samples_nested = append_conversations(samples, vqa_data, append_to="vqa")  # custom flag to append to vqa
        samples = [item for sublist in samples_nested for item in sublist]  # flatten list of lists

    # Step 4: Save result
    out_path = os.path.join(args.output_dir, f"{args.dataset.lower()}_{args.split}_frames.json")
    with open(out_path, 'w') as f:
        json.dump(samples, f, indent=2)
    print(f"[DONE] Saved {len(samples)} samples to {out_path}")



def _process_wts(caption_root, grouped, caption_lookup, args, gt_lookup):
    samples = os.listdir(caption_root)
    if "normal_trimmed" in samples:
        samples.remove("normal_trimmed")

    for sample_name in tqdm(samples, desc="WTS Samples"):
        _process_sample(sample_name, os.path.join(caption_root, sample_name), grouped, caption_lookup, args, "", gt_lookup)

    nt_path = os.path.join(caption_root, "normal_trimmed")
    if os.path.isdir(nt_path):
        for sub_sample in tqdm(os.listdir(nt_path), desc="normal_trimmed"):
            _process_sample(sub_sample, os.path.join(nt_path, sub_sample), grouped, caption_lookup, args, "normal_trimmed", gt_lookup)

def _process_bdd(caption_root, grouped, caption_lookup, args, gt_lookup):
    for root, _, files in os.walk(caption_root):
        for fname in tqdm([f for f in files if f.endswith("_caption.json")], desc="BDD Samples"):
            caption_path = os.path.join(root, fname)
            data = json.load(open(caption_path))
            sample_name = data['video_name'].replace(".mp4", "")
            video_path = os.path.join(args.video_dir, args.split, data['video_name'])
            if not os.path.exists(video_path):
                continue

            try:
                vr = VideoReader(video_path)
                fps = get_fps(video_path)
                total_frames = len(vr)
            except:
                continue

            caption_map = parse_captions_from_phase(data)

            for phase in data.get("event_phase", []):
                label = PHASE_MAP.get(str(phase["labels"][0]), f"phase_{phase['labels'][0]}")
                #frame_ids = get_centered_frames(float(phase["start_time"]), float(phase["end_time"]), fps, k=args.step_k, max_len=total_frames)
                if args.frame_strategy == "evenly":
                    start_frame = int(float(phase["start_time"]) * fps)
                    end_frame = int(float(phase["end_time"]) * fps)
                    # start_frame = math.ceil(float(phase["start_time"]) * fps)
                    # end_frame = math.floor(float(phase["end_time"]) * fps)
                    frame_ids = get_evenly_spaced_indices(start_frame, end_frame, args.num_frames)
                else:
                    frame_ids = get_centered_frames(float(phase["start_time"]), float(phase["end_time"]), fps, k=args.step_k, max_len=total_frames)                
                save_path = os.path.join(args.output_dir, args.split, sample_name, "vehicle", label)

                for fid in frame_ids:
                    frame_path = os.path.join(save_path, f"frame_{fid}.jpg")
                    if not os.path.exists(frame_path):
                        try:
                            frame = vr[fid].asnumpy()
                            os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                            if args.save:
                                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                        except:
                            continue
                    grouped[(sample_name, label, "vehicle")]["image"].append(frame_path)
                caption_lookup[(sample_name, label)] = caption_map.get(label, {})

def _process_sample(sample_name, sample_path, grouped, caption_lookup, args, sample_parent, gt_lookup):
    for view in ["overhead_view", "vehicle_view"]:
        caption_dir = os.path.join(sample_path, view)
        if not os.path.exists(caption_dir):
            continue

        for fname in os.listdir(caption_dir):
            if not fname.endswith("_caption.json"):
                continue

            data = json.load(open(os.path.join(caption_dir, fname)))
            view_folder = "overhead" if view == "overhead_view" else "vehicle"
            video_list = data.get("overhead_videos", []) if view_folder == "overhead" else [f"{sample_name}_vehicle_view.mp4"]

            caption_map = parse_captions_from_phase(data)

            for video_name in video_list:
                video_path = os.path.join(args.video_dir, args.split, sample_parent, sample_name, f"{view_folder}_view", video_name)
                if not os.path.exists(video_path):
                    continue

                try:
                    vr = VideoReader(video_path)
                    fps = get_fps(video_path)
                    total_frames = len(vr)
                except:
                    continue

                if args.dataset == 'WTS':
                    ped_bbox_file = get_bbox_file(args.bbox_dir, 'WTS', args.split, view, sample_parent, sample_name, video_name.replace(".mp4", ""), "pedestrian")
                    veh_bbox_file = get_bbox_file(args.bbox_dir, 'WTS', args.split, view, sample_parent, sample_name, video_name.replace(".mp4", ""), "vehicle")
                    ped_bboxes = load_bbox_map(ped_bbox_file)
                    veh_bboxes = load_bbox_map(veh_bbox_file)
                else:
                    ped_bboxes = veh_bboxes = {}

                for phase in data.get("event_phase", []):
                    label = PHASE_MAP.get(str(phase["labels"][0]), f"phase_{phase['labels'][0]}")
                    if args.frame_strategy == "evenly":
                        start_frame = int(float(phase["start_time"]) * fps)
                        end_frame = int(float(phase["end_time"]) * fps)
                        frame_ids = get_evenly_spaced_indices(start_frame, end_frame, args.num_frames)
                    else:
                        frame_ids = get_centered_frames(float(phase["start_time"]), float(phase["end_time"]), fps, k=args.step_k, max_len=total_frames)

                    save_path = os.path.join(args.output_dir, args.split, sample_name, view_folder, video_name.replace(".mp4", ""), label)

                    for fid in frame_ids:
                        frame_path = os.path.join(save_path, f"frame_{fid}.jpg")
                        if not os.path.exists(frame_path):
                            try:
                                frame = vr[fid].asnumpy()
                                frame = draw_bboxes(frame, ped_bboxes.get(fid, []), (255, 0, 0), "Pedestrian")
                                frame = draw_bboxes(frame, veh_bboxes.get(fid, []), (0, 0, 255), "Vehicle")
                                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                                if args.save:
                                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                            except:
                                continue
                        grouped[(sample_name, label, view_folder)]["image"].append(frame_path)
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
    parser.add_argument('--step-k', type=int, default=10)
    parser.add_argument('--frame-strategy', choices=['centered', 'evenly'], default='centered',
                    help="Frame sampling strategy: 'centered' or 'evenly'")
    parser.add_argument('--num-frames', type=int, default=5,
                    help="Number of frames to sample for 'evenly' strategy")
    parser.add_argument('--vqa-json', type=str, default=None, help="Optional VQA JSON file (only for test split)")

    args = parser.parse_args()

    extract_and_draw(args)