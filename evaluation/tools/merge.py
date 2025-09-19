import os
import glob

def merge_ranked_predictions(base_path: str, num_ranks: int, merged_path: str):
    with open(merged_path, 'w') as fout:
        for i in range(num_ranks):
            pattern = os.path.join(base_path, f"*rank{i}.jsonl")
            matched_files = glob.glob(pattern)

            if not matched_files:
                print(f"[!] Skipped (not found): {pattern}")
                continue

            file_path = matched_files[0]  # take first match
            with open(file_path, 'r') as fin:
                for line in fin:
                    fout.write(line)
            print(f"[✓] Merged: {file_path}")

    print(f"\n[✓] Final merged output written to: {merged_path}")
