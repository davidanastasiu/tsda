# Multi-Agent Cooperation for Traffic Safety Description and Analysis

We present Multi-Agent Cooperation for Traffic Safety Description and Analysis as a part of AICITY Challenge 2025 Track 2. The main goal was to generate dense traffic safety captions and VQA from multiple view points of accident scenarios. Our approach scored 2nd rank in the challenge. Please cite our work if you make use of our code and approach.

```bash
@inproceedings{kachhadiya-KachhadiyaPA25,
   author    = {Ridham Kachhadiya and Dhanishtha Patil and David C. Anastasiu},
   title     = {Multi-Agent Cooperation for Traffic Safety Description and Analysis},
   month     = {October},
   booktitle = {The International Conference on Computer Vision (ICCV) Workshops},
   year      = {2025},
}
```
## Environment Setup

### Create a new Conda environment (eg: tsda)

```bash
conda create -n tsda python=3.9.23
```
```bash
conda activate tsda
```
### Installing dependencies and libraries
```bash
cd tsda
conda env update --file environment.yml

pip install torch==2.2.0+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-build-isolation flash-attn==2.7.0.post2
```

## Dataset Structure
Prepare and arrange both the datasets inside `tsda/data/` the challenge format as below:
#### WTS
```text
WTS/
├── videos/
│   ├── train/
│   │   ├── 20230707_8_SN46_T1/
│   │   │   ├── overhead_view/
│   │   │   │   ├── 20230707_8_SN46_T1_Camera1_0.mp4
│   │   │   │   └── …                     # other views
│   │   │   └── vehicle_view/
│   │   │       └── 20230707_8_SN46_T1_vehicle_view.mp4
│   │   └── …                             # other samples
│   └── val/ …                            # same as train/
└── annotations/
    ├── bbox_generated/
    │   ├── pedestrian/
    │   │   ├── train/
    │   │   │   ├── 20230707_8_SN46_T1/
    │   │   │   │   ├── overhead_view/  … # *_bbox.json
    │   │   │   │   └── vehicle_view/  …
    │   │   │   └── …
    │   │   └── val/ …                    # same as train/
    │   └── vehicle/
    │       ├── train/
    │       │   ├── 20230707_8_SN46_T1/
    │       │   │   ├── overhead_view/  …
    │       │   │   └── vehicle_view/  …
    │       │   └── …
    │       └── val/ …                    # same as train/
    ├── vqa/
    │   ├── train/
    │   │   ├── 20230707_8_SN46_T1/
    │   │   │   ├── environment/   20230707_8_SN46_T1.json
    │   │   │   ├── overhead_view/ 20230707_8_SN46_T1_overhead_view.json
    │   │   │   └── vehicle_view/  20230707_8_SN46_T1.json
    │   │   └── …
    │   └── val/ …                        # same as train/
    └── captions/
        ├── train/
        │   ├── 20230707_8_SN46_T1/
        │   │   ├── overhead_view/
        │   │   │   └── 20230707_8_SN46_T1_caption.json
        │   │   └── vehicle_view/
        │   │       └── 20230707_8_SN46_T1_caption.json
        │   └── …                         # more samples
        └── val/ …                        # same as train/
```
#### BDD
```text
BDD_PC_5k/
├── videos/
│   ├── train/
│   │   ├── video5.mp4
│   │   ├── video9.mp4
│   │   ├── video12.mp4
│   │   └── …                         # other training samples
│   └── val/
│       └── …                         # same as train/
└── annotations/
    ├── bbox_generated/
    │   ├── train/
    │   │    ├── video5_bbox.json
    │   │    ├── video9_bbox.json
    │   │    ├── video12_bbox.json
    │   │    └── …                    # other bbox files
    │   └── val/
    │       └── …                     # same as train/
    └── caption/
        ├── train/
        │   ├── video5_caption.json
        │   ├── video9_caption.json
        │   ├── video12_caption.json
        │   └── …                    # other caption files
        └── val/
            └── …                   # test caption files
```
## Dataset Extraction & Preparation (Sub-tasks: VQA and Caption)

### Extract Frames & Annotations

Run the following script inside `tsda/extract` to extract frames with both strategies (Mid k-spaced & Evenly spaced) and annotations from both **WTS** and **BDD** datasets (train/val/test splits):

```bash
cd tsda/extract
sh extract_even_frames.sh
sh extract_mid_frames.sh
```

This will generate respective .json files for respective splits and datasets (eg. wts_train_frames.json, bdd_train_frames.json, etc.) inside the WTS/BDD frames output directory.

### Prepare datasets in InternVL3 Fine-tuning format

Run prepare_ft_data_finetune.sh to convert the dataset into required fine-tuning format for all VQA Agents.

```bash
sh prepare_ft_data_vqa.sh
```

Similarly run prepare_data_caption.sh to convert the dataset in required fine-tuning format for all Captioning Agents.
Note: Ped caption and vehicle caption are seperately finetuned. Each Agent focuses in a specific domain.

```bash
sh prepare_ft_data_caption.sh
```

You can now proceed to fine-tuning for both subtasks using the generated train and val **.jsonl** files for each specific agents inside `tsda/InternVL3_ft_data_vqa/` and `tsda/InternVL3_ft_data_caption/`.


## InternVL3 Fine-tuning for VQA and Caption Sub-tasks

### Download InternVL3 14B model once inside `tsda/base_agent/` using the following commands:

```bash
cd tsda/base_agent/

# Download OpenGVLab/InternVL3-14B
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL3-14B --local-dir InternVL3-14B
```

Now create and prepare a metadata (.json file) containing information about datasets inside `tsda/InternVL/internvl_chat/shell/data`.

Note: You can refer to official repo of InternVL for more information.

Then add or verify train and val .jsonl file paths created in previous InternVL3 finetuning format step for respective subtasks.

Eg. `mid_frames_vqa.json` (Metadata file)
```json
{
  "vlm-train": {
    "root": "tsda/InternVL3_ft_data_vqa/mid_frames_all",
    "annotation": "mid_train_all.jsonl",
    "data_augment": false,
    "max_dynamic_patch":4,
    "repeat_time": 1,
    "length": 12985
  },
  "vlm-val": {
    "root": "tsda/InternVL3_ft_data_vqa/mid_frames_all",
    "annotation": "mid_val_all.jsonl",
    "data_augment": false,
    "max_dynamic_patch": 4,
    "repeat_time": 1,
    "length": 5270
  }
}
```
Similarly, Create meta data .jsonl files for each variant.

Navigate to `tsda/InternVL/internvl_chat/shell/internvl3.0/2nd_finetune/` and configure paths or make any changes if needed in all the scripts starting with "ft_".

Note: Use the default hyper parameters for optimal training and performance.

### Training

**Hardware Configuration**: We recommend using 8xA100 GPUs for full fine-tuning of InternVL3 14B and used deepspeed for reduced memory consumption. For this pipeline, PEFT methods like LoRA doesn't provide optimal performance and scores but requires less compute resources.

Start fine-tuning by running the following script:
```bash
cd /tsda//InternVL/internvl_chat
# For VQA Mid BDD Agent
GPUS=8 PER_DEVICE_BATCH_SIZE=1 bash shell/internvl3.0/2nd_finetune/ft_vqa_mid_bdd.sh
# For VQA Mid all Agent
GPUS=8 PER_DEVICE_BATCH_SIZE=1 bash shell/internvl3.0/2nd_finetune/ft_vqa_mid_frames_all.sh
.
.
# For Caption Mid Facts Agent
GPUS=8 PER_DEVICE_BATCH_SIZE=1 bash shell/internvl3.0/2nd_finetune/ft_cap_mid_facts.sh
# For Caption Ped QA Agent
GPUS=8 PER_DEVICE_BATCH_SIZE=1 bash shell/internvl3.0/2nd_finetune/ft_cap_mid_ped_QA.sh
.
.
```
Note: Use the respective .sh script to finetune other variants, as they are self contained with their own hyperparameter configurations. 

After Fine-tuning completes, make sure to copy *.py files from `tsda/base_agent/` to saved model checkpoint for all the variants as its neccessary to perform inference.

Eg:
```bash
cp tsda/base_agent/*.py /path/to/checkpoint/mid_all_vqa/checkpoint-572
```

## VQA - Visual Question Answering (Sub-task 2)

### Validation Inference

For Validation inference on all vqa agents, run the following script:

```bash
cd inference/
sh inference_vqa_val.sh
```

### Validation Post Processing

For postprocessing on validation set, you can run the following script on all the predictions at once and results for all agents will be store inside `vqa_results_val`.

```bash
cd tsda/evaluation
sh run_postprocess_val.sh
```
Note: This will generate predictions that can be used for evaluation and multi agent selection.

### Multi Agent Cooperation 

Validation results for all the agents is used to generate multi agent collaborated results based on agents performing best on specific group of questions and the dataset split.

Following script will output `best_QA_agents_vqa.json` that contains mapping of each question with best performing agent and `multi_agent_results_vqa.json` that contains responses from all the agents.

```bash
cd tsda/multi_agent_selection
sh select_best_agents.sh
```

#### Validation Scores
```bash
cd tsda/evaluation

python evaluate_agents.py \
--pred_dir vqa_results_val \
--multi_agent_path multi_agent_selection/multi_agent_results_vqa.json \
--gt_dirs /path/to/data/WTS/vqa/val /path/to/data/BDD/vqa/val
```

### Inference, Post Processing, and Results in Submission format (Test Set)

Proceed with Inference on Test set for all the models finetuned using the saved checkpoints. Configure model checkpoint paths if needed.

Navigate to `tsda/inference/` and run the following:
```bash
cd inference/
sh inference_vqa_test.sh
```

For postprocessing on test set and converting the predictions in submission format, run the following script:

```bash
cd tsda/
sh submission/postprocess_test_all.sh
```

Test Predictions for all the agents will be processed and stored inside **vqa_results_test** and can be used for submission and generating multi agent vqa predictions.

### Multi Agent VQA predictions routing

For Multi Agent VQA generation on test set, you can run the following script after navigating inside `tsda/submission/` directory.

```bash
cd submission/
sh multi_agent_test_pred.sh
```
Multi Agent VQA results will be stored inside `vqa_results_test/multi_agent_pred` in all formats.

## Visual Captioning (Subtask-1)

### Validation Inference

```bash
cd tsda/multi_agent_selection/
sh prepare_val_inf_cap.sh
```
Inference ready .jsonl files will be saved inside `ma_val_cap_inf` and can be used for inference on all captioning agents by running the following script:

```bash
cd tsda/inference/
sh inference_cap_val.sh
```

### Validation Post Processing

Run the following script to clean and postprocess val caption results that can be used to calculate scores.

```bash
cd evaluation/
sh postprocess_cap_val_all.sh
```

Captioning agents validation results scores can be printed and evaluated using `evaluate_caption_agents.py` and best performing agents for a specific caption/dataset type can be chosen accordingly.

### Inference, Post Processing, and Results in Submission format (Test Set)

Run the following script before inference to generate inference ready files with multi agent predictions converted to facts/QA for all captioning agents.

```bash
cd submission/
sh prepare_test_inf_cap_agents.sh
```

Proceed with Inference on Test set for all the agents using the saved checkpoints. Configure model checkpoint paths if needed.

```bash
cd inference/
sh inference_cap_test.sh
```

For postprocessing on test set and converting the predictions in submission format, run the following script:

```bash
cd evaluation/
sh postprocess_cap_test_all.sh
```
The results of all captioning agents will be stored inside `cap_results_test` and can also be used to generate multi agent test caption predictions using the following script:

```bash
python evaluation/multi_agent_cap_test.py \
--root cap_results_test \
--out cap_results_test/multi_agent_cap_test.json
```