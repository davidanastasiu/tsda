import os
import json
import math
import cv2
from collections import defaultdict
from decord import VideoReader

def get_centered_frames(start_time, end_time, fps, k=10, max_len=None):
    sid = int(math.floor(start_time * fps))
    eid = int(math.ceil(end_time * fps))
    if max_len is not None:
        eid = min(eid, max_len)
    mid = sid + (eid - sid) // 2

    # Case 1: centered 3-frame pattern
    if mid - k >= sid and mid + k < eid:
        return [mid - k, mid, mid + k]
    
    # Case 2: fallback 2-frame near end
    elif eid - k >= sid:
        return sorted(list({eid - k - 1, eid - 1}))
    
    # Case 3: fallback 2-frame near start
    elif sid + k < eid:
        return sorted(list({sid, sid + k}))

    # Case 4: final fallback - select valid ones only
    fallback = [sid, eid - 1]
    if max_len is not None:
        fallback = [f for f in fallback if 0 <= f < max_len]
    if not fallback and max_len:  # nothing valid? pick last frame
        fallback = [max_len - 1]
    return sorted(set(fallback))

def get_evenly_spaced_indices(start_frame, end_frame, num_frames):
    if end_frame <= start_frame or num_frames == 0:
        return []
    total = end_frame - start_frame
    step = max(total // num_frames, 1)
    return [start_frame + i * step for i in range(num_frames)]

def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30.0

def load_bbox_map(bbox_json):
    if not os.path.exists(bbox_json):
        return {}
    try:
        with open(bbox_json) as f:
            data = json.load(f)
        bbox_map = defaultdict(list)
        for ann in data.get("annotations", []):
            raw_id = ann.get("image_id")
            bbox = ann.get("bbox")
            if raw_id is None or bbox is None:
                continue
            if isinstance(raw_id, str) and raw_id[-6:].isdigit():
                frame_id = int(raw_id[-6:])
            else:
                frame_id = raw_id
            bbox_map[frame_id].append(bbox)
        return bbox_map
    except Exception as e:
        print(f"[ERROR] Failed to load bbox file {bbox_json}: {e}")
        return {}

def draw_bboxes(frame, bbox_list, color, label_text=None):
    if bbox_list:
        bbox = bbox_list[0]
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        if label_text:
            cv2.putText(frame, label_text, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame

def get_bbox_file(bbox_root, dataset, split, view_type, sample_parent, sample_name, video_name, bbox_type):
    subfolder = os.path.join(bbox_root, bbox_type, split, sample_parent, sample_name, view_type)
    bbox_file = os.path.join(subfolder, f"{video_name}_bbox.json")
    return bbox_file


PHASE_MAP = {
    '0': 'prerecognition',
    '1': 'recognition',
    '2': 'judgement',
    '3': 'action',
    '4': 'avoidance',
    'prerecognition': 'prerecognition',
    'recognition': 'recognition',
    'judgement': 'judgement',
    'action': 'action',
    'avoidance': 'avoidance'
}
