# import json
# from pathlib import Path
# from typing import Dict, List, Tuple

# def parse_captions_from_phase(data):
#     """
#     Given loaded JSON with 'event_phase', return dict[label] -> captions
#     """
#     caption_map = {}
#     for phase in data.get("event_phase", []):
#         label = str(phase["labels"][0])
#         caption_map[label] = {
#             "caption_pedestrian": phase.get("caption_pedestrian", ""),
#             "caption_vehicle": phase.get("caption_vehicle", "")
#         }
#     return caption_map


# def load_captions_from_dirs(root_dirs: List[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
#     """
#     Return dict: (sample_name, label) -> {"caption_pedestrian": ..., "caption_vehicle": ...}
#     """
#     caption_lookup = {}

#     for root_dir in root_dirs:
#         root_dir = Path(root_dir)
#         for sample_dir in root_dir.rglob("*"):
#             if not sample_dir.is_dir():
#                 continue

#             sample_name = sample_dir.name

#             for view in ["overhead_view", "vehicle_view"]:
#                 caption_path = sample_dir / view / f"{sample_name}_caption.json"
#                 if not caption_path.exists():
#                     continue

#                 try:
#                     data = json.load(open(caption_path))
#                 except Exception as e:
#                     print(f"[WARN] Failed to read caption {caption_path}: {e}")
#                     continue

#                 phase_map = parse_captions_from_phase(data[0]) if isinstance(data, list) else parse_captions_from_phase(data)
#                 for label, cap in phase_map.items():
#                     caption_lookup[(sample_name, label)] = cap
#     return caption_lookup

import json
from pathlib import Path
from typing import Dict, List, Tuple
from extract.utils import PHASE_MAP  # make sure this is imported

def parse_captions_from_phase(data):
    """
    Given loaded JSON with 'event_phase', return dict[label] -> captions
    """
    caption_map = {}
    for phase in data.get("event_phase", []):
        label = PHASE_MAP.get(str(phase["labels"][0]), f"phase_{phase['labels'][0]}")
        caption_map[label] = {
            "caption_pedestrian": phase.get("caption_pedestrian", ""),
            "caption_vehicle": phase.get("caption_vehicle", "")
        }
    return caption_map


def load_captions_from_dirs(root_dirs: List[str]) -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Return dict: (sample_name, label) -> {"caption_pedestrian": ..., "caption_vehicle": ...}
    """
    caption_lookup = {}

    for root_dir in root_dirs:
        root_dir = Path(root_dir)
        for sample_dir in root_dir.rglob("*"):
            if not sample_dir.is_dir():
                continue

            sample_name = sample_dir.name

            for view in ["overhead_view", "vehicle_view"]:
                caption_path = sample_dir / view / f"{sample_name}_caption.json"
                if not caption_path.exists():
                    continue

                try:
                    data = json.load(open(caption_path))
                except Exception as e:
                    print(f"[WARN] Failed to read caption {caption_path}: {e}")
                    continue

                phase_map = parse_captions_from_phase(data[0]) if isinstance(data, list) else parse_captions_from_phase(data)
                for label, cap in phase_map.items():
                    caption_lookup[(sample_name, label)] = cap
    return caption_lookup
