
cd "$(dirname "$0")/.."

# Path Configuration:

DATA_DIR="data"


VQA_RESULTS_DIR="vqa_results_test"
OUTPUT_DIR="ma_test_cap_inf"

# Input frame directories (now located directly inside TSDA)
WTS_MID_FRAMES_DIR="WTS_mid_frames"
BDD_MID_FRAMES_DIR="BDD_mid_frames"
WTS_EVENLY_FRAMES_DIR="WTS_even_frames"
BDD_EVENLY_FRAMES_DIR="BDD_even_frames"


MULTIAGENT_VQA_JSONL="$VQA_RESULTS_DIR/multi_agent_pred/multi_agent_test_pred.jsonl"
TEST_FULL_REF_JSONL="$VQA_RESULTS_DIR/mid_all_e3/merged.jsonl"
VIEW_CSV="$DATA_DIR/WTS/view_used_as_main_reference_for_multiview_scenario.csv"


echo "Creating output directory at: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "--------------------------------------------"


echo "Processing caption facts..."
python -m submission.prepare_test_inf_cap_agents \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_test_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_test_frames.json" \
    --multiagent-vqa-jsonl "$MULTIAGENT_VQA_JSONL" \
    --test-full-ref-jsonl "$TEST_FULL_REF_JSONL" \
    --format caption_facts \
    --view-csv "$VIEW_CSV" \
    --out-jsonl "$OUTPUT_DIR/ma_test_caption_facts.jsonl"


echo "Processing selective QA..."
python -m submission.prepare_test_inf_cap_agents \
    --wts-json "$WTS_MID_FRAMES_DIR/wts_test_frames.json" \
    --bdd-json "$BDD_MID_FRAMES_DIR/bdd_test_frames.json" \
    --multiagent-vqa-jsonl "$MULTIAGENT_VQA_JSONL" \
    --test-full-ref-jsonl "$TEST_FULL_REF_JSONL" \
    --view-csv "$VIEW_CSV" \
    --format selective_qa \
    --out-ped-qa-jsonl "$OUTPUT_DIR/ma_test_ped_qa.jsonl" \
    --out-veh-qa-jsonl "$OUTPUT_DIR/ma_test_veh_qa.jsonl"


echo "Processing evenly-spaced frame QA..."
python -m submission.prepare_test_inf_cap_agents \
    --wts-json "$WTS_EVENLY_FRAMES_DIR/wts_test_frames.json" \
    --bdd-json "$BDD_EVENLY_FRAMES_DIR/bdd_test_frames.json" \
    --multiagent-vqa-jsonl "$MULTIAGENT_VQA_JSONL" \
    --test-full-ref-jsonl "$TEST_FULL_REF_JSONL" \
    --format even_qa \
    --out-jsonl "$OUTPUT_DIR/ma_test_even_qa.jsonl"

echo "Test inference data prepared for captioning agents based on multi agent VQA predictions."