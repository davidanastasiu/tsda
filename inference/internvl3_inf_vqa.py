import argparse
import json
import os
import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import logging
import warnings

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")

# --- Constants ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Default value, can be overridden by CLI arg
MAX_TOTAL_TOKENS = 6144
TILE_TOKEN_ESTIMATE = 256

# --- Image Preprocessing ---
def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=True):
    """Dynamically preprocesses an image by splitting it into tiles based on aspect ratio."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    def find_closest_ratio():
        best_ratio = (1, 1)
        min_diff = float('inf')
        for r in target_ratios:
            diff = abs(aspect_ratio - r[0] / r[1])
            if diff < min_diff:
                min_diff = diff
                best_ratio = r
        return best_ratio

    r_w, r_h = find_closest_ratio()
    target_width, target_height = r_w * image_size, r_h * image_size
    resized = image.resize((target_width, target_height))
    processed = []

    for i in range(r_h):
        for j in range(r_w):
            box = (j * image_size, i * image_size, (j + 1) * image_size, (i + 1) * image_size)
            processed.append(resized.crop(box))

    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_image_tensor(image_paths, tokenizer, prompt_text, input_size=448, max_token_limit=MAX_TOTAL_TOKENS, tile_token_estimate=TILE_TOKEN_ESTIMATE):
    """Loads images from paths, preprocesses them, and returns a tensor, respecting token limits."""
    transform = build_transform(input_size)
    all_tiles = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True)
        all_tiles.extend([transform(t) for t in tiles])

    prompt_tokens = len(tokenizer(prompt_text)["input_ids"])
    # Ensure there is at least one tile
    max_tiles = max((max_token_limit - prompt_tokens) // tile_token_estimate, 1)
    all_tiles = all_tiles[:max_tiles]

    return torch.stack(all_tiles)

# --- Inference ---
def run_inference(args):
    """Initializes distributed process, loads model, and runs inference on a chunk of data."""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    logging.info(f"Rank {local_rank}/{world_size} starting...")

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map={"": torch.device(f"cuda:{local_rank}")}
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1, do_sample=False)

    with open(args.input_jsonl, 'r') as f:
        all_lines = f.readlines()

    total = len(all_lines)
    chunk_size = total // world_size
    start = local_rank * chunk_size
    end = total if local_rank == world_size - 1 else (local_rank + 1) * chunk_size
    lines = all_lines[start:end]

    logging.info(f"Rank {local_rank} processing {len(lines)} samples: [{start}, {end})")

    results = []
    num_skipped = 0

    for idx, line in enumerate(tqdm(lines, desc=f"Rank {local_rank}")):
        try:
            entry = json.loads(line.strip())
            image_paths = entry["image"]
            conv_pairs = entry.get("conversations", [])
            sample_id = entry.get("id", f"rank{local_rank}_idx{idx}")
            label = entry.get("meta", {}).get("label", "")

            if not conv_pairs:
                logging.warning(f"[Rank {local_rank}] Skipped {sample_id}: no conversations.")
                num_skipped += 1
                continue

            pixel_values = load_image_tensor(
                image_paths,
                tokenizer,
                conv_pairs[0]["value"],
                input_size=args.img_size,
                max_token_limit=args.max_total_tokens
            )
            pixel_values = pixel_values.to(torch.bfloat16).to(f"cuda:{local_rank}")

            sample_result = {
                "meta": entry.get("meta", {}),
                "predictions": [],
                "label": label
            }
            for i in range(0, len(conv_pairs), 2):
                prompt = conv_pairs[i]["value"]
                response = model.chat(tokenizer, pixel_values, prompt, generation_config)
                sample_result["predictions"].append({"question": prompt, "answer": response})

            results.append(sample_result)

        except Exception as e:
            logging.error(f"[Rank {local_rank}] Error in sample idx {idx}: {e}")
            continue

    out_path = args.output_jsonl.replace(".jsonl", f"_rank{local_rank}.jsonl")
    with open(out_path, 'w') as fout:
        for r in results:
            fout.write(json.dumps(r) + "\n")

    logging.info(f"[\u2713] Rank {local_rank} finished. Saved {len(results)} samples to {out_path}. Skipped: {num_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Base output JSONL file (suffix _rankX will be added)")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned InternVL3 model")
    parser.add_argument("--img-size", type=int, default=448, help="Image size for preprocessing")
    parser.add_argument("--max-total-tokens", type=int, default=MAX_TOTAL_TOKENS, help="Maximum total tokens for context")
    args = parser.parse_args()
    run_inference(args)
