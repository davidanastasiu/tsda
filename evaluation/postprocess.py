import argparse
from evaluation.tools.merge import merge_ranked_predictions
from evaluation.tools.clean import trim_predictions
from evaluation.tools.convert import load_gt_from_val_dirs, convert_predictions

def run_pipeline(args):
    merged_file = args.merged_jsonl or f"{args.base_path}/merged.jsonl"
    trimmed_file = args.trimmed_jsonl or f"{args.base_path}/trimmed.jsonl"
    converted_file = args.converted_json or f"{args.base_path}/converted.json"

    if args.merge:
        merge_ranked_predictions(args.base_path, args.num_ranks, merged_file)
    if args.trim:
        trim_predictions(merged_file, trimmed_file)
    if args.convert:
        gt_dict = load_gt_from_val_dirs(args.gt_dirs)
        convert_predictions(trimmed_file, gt_dict, converted_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True, help="Base folder with rank*.jsonl files")
    parser.add_argument("--num_ranks", type=int, default=8, help="Number of ranks to merge")
    parser.add_argument("--gt_dirs", type=str, nargs="+", help="List of GT root dirs")
    parser.add_argument("--merged_jsonl", type=str, help="Path to save merged .jsonl file")
    parser.add_argument("--trimmed_jsonl", type=str, help="Path to save trimmed .jsonl file")
    parser.add_argument("--converted_json", type=str, help="Path to save converted .json file")
    parser.add_argument("--merge", action="store_true", help="Run merging step")
    parser.add_argument("--trim", action="store_true", help="Run prompt cleanup step")
    parser.add_argument("--convert", action="store_true", help="Run conversion step")

    args = parser.parse_args()
    run_pipeline(args)
