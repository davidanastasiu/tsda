"""
Evaluate validation set result for the AI City Challenge, Track 2, 2024.
"""

import os
import glob
import json
import utils
import warnings
import traceback

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(add_help=False, usage=usage_msg())
    parser.add_argument('--help', action='help', help='Show this help message and exit')
    parser.add_argument("--predictions_file", type=str, help="path to predictions file", required=True)
    parser.add_argument("--ground_truth", type=str, default="labels/gt", help="path to ground truth json files")
    return parser.parse_args()


def usage_msg():
    return """ 
    python3 metrics_all.py --predictions_file <path_to_predictions_json> --ground_truth <path_to_ground_truth_folder>
    See python3 metrics_all.py --help for more info.
    """


def usage(msg=None):
    if msg:
        print("%s\n" % msg)
    print("\nUsage: %s" % usage_msg())
    exit()


def read_pred(pred_json_path):
    with open(pred_json_path) as f:
        data = json.load(f)
    return data


def read_gt_one_scenario(gt_json_path):
    with open(gt_json_path) as f:
        data = json.load(f)
    return data["event_phase"]


def read_gt(gt_dir_path):
    gt_annotations = {}
    for file_path in glob.iglob(gt_dir_path + '/**/**.json', recursive=True):
        if "vehicle_view" in file_path:
            continue
        file_name = file_path.split("/")[-1]
        scenario_name = file_name.strip("_caption.json")
        gt_annotation = read_gt_one_scenario(file_path)
        gt_annotations[scenario_name] = gt_annotation
    return gt_annotations

def convert_segments_to_dict(predictions_dict):
    """
    Convert segment lists to dicts with stringified indices.
    Input: {
      "scene_1": [ {...}, {...} ],
      ...
    }
    Output: {
      "scene_1": {
        "0": {...},
        "1": {...}
      },
      ...
    }
    """
    converted = {}
    for scene, segments in predictions_dict.items():
        if isinstance(segments, list):
            converted[scene] = {
                str(i): segment for i, segment in enumerate(segments)
            }
        else:
            converted[scene] = segments  # already dict
    return converted


def compute_metrics_scenario(pred_scenario: list, gt_scenario: list, scenario_name: str):
    pred_scenario_dict = utils.convert_to_dict(pred_scenario)
    gt_scenario_dict = utils.convert_to_dict(gt_scenario)

    metrics_ped_scenario_total = {"bleu": 0, "meteor": 0, "rouge-l": 0, "cider": 0}
    metrics_veh_scenario_total = {"bleu": 0, "meteor": 0, "rouge-l": 0, "cider": 0}
    num_segments = 0

    for segment, gt_segment_dict in gt_scenario_dict.items():
        if segment not in pred_scenario_dict:
            print(f"Segment captions missing for scenario {scenario_name}, segment number {segment}")
            num_segments += 1
            continue

        pred_segment_dict = pred_scenario_dict[segment]

        metrics_ped_segment_total = utils.compute_metrics_single(
            pred_segment_dict["caption_pedestrian"], gt_segment_dict["caption_pedestrian"])
        metrics_veh_segment_total = utils.compute_metrics_single(
            pred_segment_dict["caption_vehicle"], gt_segment_dict["caption_vehicle"])

        for metric_name, metric_score in metrics_ped_segment_total.items():
            metrics_ped_scenario_total[metric_name] += metric_score
        for metric_name, metric_score in metrics_veh_segment_total.items():
            metrics_veh_scenario_total[metric_name] += metric_score

        num_segments += 1

    return metrics_ped_scenario_total, metrics_veh_scenario_total, num_segments


def compute_metrics_overall(pred_all, gt_all):
    metrics_pedestrian_overall = {"bleu": 0, "meteor": 0, "rouge-l": 0, "cider": 0}
    metrics_vehicle_overall = {"bleu": 0, "meteor": 0, "rouge-l": 0, "cider": 0}
    num_segments_overall = 0

    for scenario_name, gt_scenario in gt_all.items():
        if scenario_name not in pred_all:
            print(f"Scenario {scenario_name} exists in ground-truth but not in predictions. Counting zero score.")
            num_segments = len(gt_scenario)
            num_segments_overall += num_segments
            continue

        pred_scenario = pred_all[scenario_name]

        metrics_ped_scenario_total, metrics_veh_scenario_total, num_segments = compute_metrics_scenario(
            pred_scenario, gt_scenario, scenario_name)

        for metric_name, metric_score in metrics_ped_scenario_total.items():
            metrics_pedestrian_overall[metric_name] += metric_score
        for metric_name, metric_score in metrics_veh_scenario_total.items():
            metrics_vehicle_overall[metric_name] += metric_score

        num_segments_overall += num_segments

    return metrics_pedestrian_overall, metrics_vehicle_overall, num_segments_overall


def compute_mean_metrics(metrics_overall, num_segments_overall):
    return {k: v / num_segments_overall for k, v in metrics_overall.items()}


def print_metrics(metrics_dict):
    for metric_name, metric_val in metrics_dict.items():
        print(f"- {metric_name}: {metric_val:.3f}")


def filter_internal_or_external_data(data, internal):
    return {
        key: value for key, value in data.items()
        if (internal and key.startswith("2023")) or (not internal and key.startswith("video"))
    }


def evaluate_one_dataset(predictions_file, ground_truth_dir_path, internal):
    try:
        pred_all = read_pred(predictions_file)
        #pred_all = convert_segments_to_dict(pred_all)
        if isinstance(list(pred_all.values())[0], list):
            pred_all = {
                scene: {
                    str(i): seg for i, seg in enumerate(segments)
                } 
                for scene, segments in pred_all.items()
            }
        gt_all = read_gt(ground_truth_dir_path)
        pred_all = filter_internal_or_external_data(pred_all, internal)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            metrics_pedestrian_overall, metrics_vehicle_overall, num_segments_overall = compute_metrics_overall(
                pred_all, gt_all)

            metrics_pedestrian_mean = compute_mean_metrics(metrics_pedestrian_overall, num_segments_overall)
            metrics_vehicle_mean = compute_mean_metrics(metrics_vehicle_overall, num_segments_overall)

        # Compute per-metric averages between ped and veh
        metrics_all_category_mean = {
            metric: (metrics_pedestrian_mean[metric] + metrics_vehicle_mean[metric]) / 2
            for metric in metrics_pedestrian_mean
        }

        # Compute final score correctly based on averaged metrics
        mean_score = (
            metrics_all_category_mean["bleu"] * 100 +
            metrics_all_category_mean["meteor"] * 100 +
            metrics_all_category_mean["rouge-l"] * 100 +
            metrics_all_category_mean["cider"] * 10
        ) / 4

        # Compute individual ped/veh scores
        total_ped = (
            metrics_pedestrian_mean["bleu"] * 100 +
            metrics_pedestrian_mean["meteor"] * 100 +
            metrics_pedestrian_mean["rouge-l"] * 100 +
            metrics_pedestrian_mean["cider"] * 10
        )/4
        total_veh = (
            metrics_vehicle_mean["bleu"] * 100 +
            metrics_vehicle_mean["meteor"] * 100 +
            metrics_vehicle_mean["rouge-l"] * 100 +
            metrics_vehicle_mean["cider"] * 10
        )/4

        print(f"=== Results for {'internal' if internal else 'external'} videos ===")
        print(f"Pedestrian mean score over all data provided:")
        print_metrics(metrics_pedestrian_mean)
        print(f"Vehicle mean score over all data provided:")
        print_metrics(metrics_vehicle_mean)
        print(f"Pedestrian total score (range [0, 100]): {total_ped:.2f}")
        print(f"Vehicle total score (range [0, 100]): {total_veh:.2f}")
        print(f"mean score (range [0, 100]): {mean_score:.2f}")
        print("==" * 20)

    except Exception as e:
        print("Error: %s" % repr(e))
        traceback.print_exc()
        exit()

    return metrics_all_category_mean, mean_score


if __name__ == '__main__':
    args = get_args()

    # gt_internal = f'{args.ground_truth}/annotations'
    # gt_external = f'{args.ground_truth}/external/BDD_PC_5K/annotations'
    gt_internal = f'{args.ground_truth}/WTS/annotations/caption/val'
    gt_external = f'{args.ground_truth}/BDD_PC_5k/annotations/caption/val'
    if not os.path.exists(gt_internal) or not os.path.exists(gt_external):
        print("Error: Internal or external ground truth labels missing.")
        exit()

    metrics_all_category_mean_internal, mean_score_internal = evaluate_one_dataset(
        args.predictions_file, gt_internal, internal=True)

    metrics_all_category_mean_external, mean_score_external = evaluate_one_dataset(
        args.predictions_file, gt_external, internal=False)

    final_score_overall = (mean_score_internal + mean_score_external) / 2

    results = {f'{k}_i': v for k, v in metrics_all_category_mean_internal.items()}
    results.update({f'{k}_e': v for k, v in metrics_all_category_mean_external.items()})
    results['s2'] = final_score_overall

    print("Final mean score: " + str(final_score_overall))