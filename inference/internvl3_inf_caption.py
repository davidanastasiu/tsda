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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
warnings.filterwarnings("ignore", message="Setting pad_token_id to eos_token_id")


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_TOTAL_TOKENS = 6144
TILE_TOKEN_ESTIMATE = 256

# -----------------------
# Image Preprocessing
# -----------------------
def build_transform(input_size: int):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image: Image.Image, min_num=1, max_num=12, image_size=336, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
         for j in range(1, n + 1) if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1]
    )

    def closest_ratio():
        best = (1, 1); best_diff = float('inf')
        for r in target_ratios:
            diff = abs(aspect_ratio - r[0] / r[1])
            if diff < best_diff:
                best, best_diff = r, diff
        return best

    r_w, r_h = closest_ratio()
    target_width, target_height = r_w * image_size, r_h * image_size
    resized = image.resize((target_width, target_height))
    tiles = []
    for i in range(r_h):
        for j in range(r_w):
            box = (j * image_size, i * image_size, (j + 1) * image_size, (i + 1) * image_size)
            tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles

def load_image_tensor(image_paths, tokenizer, prompt_text, input_size=336,
                      max_token_limit=MAX_TOTAL_TOKENS, tile_token_estimate=TILE_TOKEN_ESTIMATE):
    transform = build_transform(input_size)
    all_tiles = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True)
        all_tiles.extend([transform(t) for t in tiles])

    prompt_tokens = len(tokenizer(prompt_text)["input_ids"])
    max_tiles = max((max_token_limit - prompt_tokens) // tile_token_estimate, 1)
    all_tiles = all_tiles[:max_tiles]
    return torch.stack(all_tiles)


# Inference helpers

def _single_turn(rank, device, model, tokenizer, lines, out_path, image_size, max_new_tokens):
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    results, skipped = [], 0

    for idx, line in enumerate(tqdm(lines, desc=f"Rank {rank} / single")):
        try:
            entry = json.loads(line.strip())
            image_paths = entry.get("image", [])
            conv = entry.get("conversations", [])
            sample_id = entry.get("meta", {}).get("sample", f"rank{rank}_idx{idx}")

            if not image_paths or not conv or "value" not in conv[0]:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: missing image or prompt.")
                skipped += 1
                continue

            prompt = conv[0]["value"]
            tiles = load_image_tensor(image_paths, tokenizer, prompt, input_size=image_size).to(torch.bfloat16).to(device)
            reply = model.chat(tokenizer, tiles, prompt, gen_cfg).strip()

            if len(conv) > 1 and conv[1].get("from") == "gpt":
                conv[1]["value"] = reply
            else:
                entry.setdefault("conversations", []).append({"from": "gpt", "value": reply})

            results.append(entry)

        except Exception as e:
            logging.error(f"[Rank {rank}] Error idx {idx}: {e}")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logging.info(f"[✓] Rank {rank} single saved {len(results)} (skipped {skipped}) → {out_path}")


def _frames_facts_two_turns(rank, device, model, tokenizer, lines, out_path, image_size, max_new_tokens):
    """
    Expect EXACTLY TWO human turns per sample (facts over frames).
      [0] human Q1 → [1] gpt (overwrite/insert)
      [2] human Q2 → [3] gpt (overwrite/insert)
    """
    gen_cfg = dict(max_new_tokens=max_new_tokens, do_sample=False)
    results, skipped = [], 0

    for idx, line in enumerate(tqdm(lines, desc=f"Rank {rank} / frames_facts")):
        try:
            entry = json.loads(line.strip())
            image_paths = entry.get("image", [])
            conv = entry.get("conversations", [])
            sample_id = entry.get("meta", {}).get("sample", f"rank{rank}_idx{idx}")

            if not image_paths or not conv:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: missing images or conversations.")
                skipped += 1
                continue

            # Identify first two human turns
            human_indices = [i for i, t in enumerate(conv) if t.get("from") == "human"]
            if len(human_indices) < 2:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: expected 2 human turns, found {len(human_indices)}.")
                skipped += 1
                continue

            h1, h2 = human_indices[:2]
            prompt1 = conv[h1].get("value", "")
            if not prompt1:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: empty first human prompt.")
                skipped += 1
                continue

            # Preload tiles once
            tiles = load_image_tensor(image_paths, tokenizer, prompt1, input_size=image_size).to(torch.bfloat16).to(device)

            reply1 = model.chat(tokenizer, tiles, prompt1, gen_cfg).strip()
            if h1 + 1 < len(conv) and conv[h1 + 1].get("from") == "gpt":
                conv[h1 + 1]["value"] = reply1
            else:
                conv.insert(h1 + 1, {"from": "gpt", "value": reply1})

            next_humans = [i for i, t in enumerate(conv) if t.get("from") == "human" and i > h1 + 1]
            if not next_humans:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: could not locate second human after insertion.")
                skipped += 1
                continue

            h2_new = next_humans[0]
            prompt2 = conv[h2_new].get("value", "")
            if not prompt2:
                logging.warning(f"[Rank {rank}] Skipped {sample_id}: empty second human prompt.")
                skipped += 1
                continue

            reply2 = model.chat(tokenizer, tiles, prompt2, gen_cfg).strip()
            if h2_new + 1 < len(conv) and conv[h2_new + 1].get("from") == "gpt":
                conv[h2_new + 1]["value"] = reply2
            else:
                conv.insert(h2_new + 1, {"from": "gpt", "value": reply2})

            entry["conversations"] = conv
            results.append(entry)

        except Exception as e:
            logging.error(f"[Rank {rank}] Error idx {idx}: {e}")

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    logging.info(f"[✓] Rank {rank} frames_facts saved {len(results)} (skipped {skipped}) → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference for ped/veh/even_qa/frames_facts with a single model path (DDP).")
    parser.add_argument("--input-jsonl", required=True, help="Path to input JSONL file")
    parser.add_argument("--output-jsonl", required=True, help="Base output JSONL (suffix _rankX added)")
    parser.add_argument("--model-path", required=True, help="Path to the model to load")
    parser.add_argument("--model-type", choices=["ped", "veh", "frames_facts", "even_qa"], required=True,
                        help="Select behavior: ped/veh=single-turn; frames_facts=2-turn facts; even_qa=single-turn.")
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--use-flash-attn", action="store_true")
    args = parser.parse_args()

   
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    logging.info(f"Rank {local_rank}/{world_size} starting on {device}...")

    
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
        device_map={"": device}
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    
    with open(args.input_jsonl, "r") as f:
        all_lines = f.readlines()
    total = len(all_lines)
    chunk = total // world_size
    start = local_rank * chunk
    end = total if local_rank == world_size - 1 else (local_rank + 1) * chunk
    lines = all_lines[start:end]
    logging.info(f"Rank {local_rank} processing {len(lines)} samples: [{start}, {end})")

    out_path = args.output_jsonl.replace(".jsonl", f"_rank{local_rank}.jsonl")

    if args.model_type in ("ped", "veh", "even_qa"):
        _single_turn(local_rank, device, model, tokenizer, lines, out_path, args.image_size, args.max_new_tokens)
    elif args.model_type == "frames_facts":
        _frames_facts_two_turns(local_rank, device, model, tokenizer, lines, out_path, args.image_size, args.max_new_tokens)

    logging.info(f"[✓] Rank {local_rank} finished.")

if __name__ == "__main__":
    main()
