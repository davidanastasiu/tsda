import re
import json

def clean_question_text(q):
    q = re.sub(r'(<image>\s*)+', '', q)
    q = re.sub(r"If red and blue boxes.*?(?=(What|Which|How|Where|Who|When|Are|Is))", '', q, flags=re.DOTALL)
    q = re.sub(r'Answer with the option.*$', '', q, flags=re.IGNORECASE)
    lines = [line.strip() for line in q.strip().splitlines()]
    question_lines = []
    for line in lines:
        if re.match(r"^[A-Da-d]\.", line):
            break
        question_lines.append(line)
    for line in reversed(question_lines):
        if re.match(r"^(What|Which|How|Where|Who|When|Are|Is)\b", line, re.IGNORECASE):
            return line.strip()
    return question_lines[-1].strip() if question_lines else ""

def trim_predictions(input_path: str, output_path: str):
    missing_questions = []
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            entry = json.loads(line)
            for pred in entry.get("predictions", []):
                original = pred.get("question", "")
                cleaned = clean_question_text(original)
                pred["question"] = cleaned
                if not cleaned:
                    missing_questions.append(original[:200])
            outfile.write(json.dumps(entry) + "\n")

    if missing_questions:
        print(f"⚠️ {len(missing_questions)} predictions had empty questions:")
        for q in missing_questions[:5]:
            print("Example:\n", q.replace("\n", " "), "\n...")
    print(f"✅ Cleaned predictions written to {output_path}")
