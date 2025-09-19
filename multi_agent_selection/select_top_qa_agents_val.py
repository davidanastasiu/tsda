#!/usr/bin/python
"""
Given a set of validation results from the QA models, choose the best performing agents
for each question and compute the overall accuracy.
Choose the best performing agent based on the accuracy of the answers for each question
for the external dataset (BDD) and the internal one (WTS).
"""
import os
import json
from collections import defaultdict
from argparse import ArgumentParser
from evaluation.tools.convert import load_gt_from_val_dirs


def qa_accuracy(predictions, gt, compute_question_accuracies=True):
    # Compute accuracy
    if not isinstance(predictions, dict):

        predictions = {
            entry['id']: entry['correct'].lower()
            for entry in predictions if entry['correct'] is not None
        }


    if compute_question_accuracies:
        question_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    else:
        question_accuracy = None
    
    total = len(gt)
    correct = 0
    
    for qid, gt_answer in gt.items():
        user_answer = predictions.get(qid)

        if user_answer == gt_answer['correct'].lower():
            correct += 1
            if compute_question_accuracies:
                question_accuracy[gt_answer['question']]['correct'] += 1
        if compute_question_accuracies:
            question_accuracy[gt_answer['question']]['total'] += 1
    accuracy = 100*(correct / total) if total > 0 else 0.0
    if compute_question_accuracies:
        for _,v in question_accuracy.items():
            v['accuracy'] = 100 * (v['correct'] / v['total']) if v['total'] > 0 else 0.0
    return accuracy, question_accuracy

def get_question_subset(question, questions, internal=True):
    """
    Get the subset of predictions for a given question.
    """
    subset = []
    for _,q in questions.items():
        if q['question'] == question and q['internal'] == internal:
            subset.append(q['id'])
    return subset

def choose_best_question_model(predictions_dict, question, questions, internal=True):
    """
    Choose the best model for a given question based on the accuracy of the answers.
    """
    subset = get_question_subset(question, questions, internal)
    if not subset:
        return None, 0.0

    best_model = None
    best_accuracy = 0.0

    gt_subset = {qid: questions[qid] for qid in subset}
    for model, preds in predictions_dict.items():
        # get subset of predictions for the questions in subset
        preds_subset = {qid: preds[qid] for qid in subset if qid in preds}
        if not preds_subset:
            continue
            #raise ValueError(f"No predictions found for question {question} in model {model}")
        
        # compute accuracy
        accuracy = qa_accuracy(preds_subset, gt_subset, compute_question_accuracies=False)[0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
    return best_model, best_accuracy

def augment_gt_with_internal_flag(gt_dict):
    questions = {}
    for qid, meta in gt_dict.items():
        sample = meta["sample"]
        view = meta.get("view", "")
        internal = not sample.startswith("video")  # True for WTS, False for BDD
        questions[qid] = {
            "id": qid,
            "question": meta["question"],
            "correct": meta["correct"],
            "internal": internal,
            "view": view,
            "sample": sample,
            "label": meta.get("label", "")
        }
    return questions

def choose_best_models(predictions, questions):
    """
    Choose the best models for each question based on the accuracy of the answers.
    """
    best_models = {}
    # transform predictions to a dictionary of dictionaries
    predictions_dict = {model: {q['id']: q['correct'].lower() for q in preds} for model, preds in predictions.items()}

    # get set of questions
    question_set = set(q['question'] for q in questions.values())

    answers = {}

    for qid, q in questions.items():
        if q['question'] not in question_set:
            continue
        # Choose best model for the question
        internal = q['internal']
        k = f"{q['question']}_{internal}"
        if k in best_models:
            best_model, best_accuracy = best_models[k]
        else:
            best_model, best_accuracy = choose_best_question_model(predictions_dict, q['question'], questions, internal)
            best_models[k] = (best_model, best_accuracy)
        ans = None
        try:
            ans = predictions_dict[best_model][qid]
        except KeyError:
            print(f"Warning: No predictions found for question {q['id']} in model {best_model}")
        if ans is None:
            # find an alternative model that has predictions for this question
            for model, preds in predictions_dict.items():
                if qid in preds:
                    ans = preds[qid]
                    break
        if ans is None:
            print(f"Error: No predictions found for question {q['id']} in any model")
            continue
        answers[qid] = {
            'id': qid,
            'correct': ans
        }
    # Return the answers and the best models
    print(f"✅ Final predictions generated: {len(answers)} questions")

    return list(answers.values()), best_models

def load_prediction_models(models_path):
    """
    Load the prediction models from the given path.
    """
    predictions = {}
    if os.path.isdir(models_path):
        # If the path is a directory, load all JSON files in it
        for filename in os.listdir(models_path):
            if filename.endswith('.json'):
                model_path = os.path.join(models_path, filename)
                with open(model_path, 'r') as f:
                    try:
                        predictions[filename] = json.load(f)  # Load the predictions for the model
                    except json.JSONDecodeError:
                        print(f"Error: {model_path} is not a valid JSON file.")
                        continue
    elif ',' in models_path:
        for model in models_path.split(','):
            model = model.strip()
            if not model.endswith('.json'):
                model += '.json'
            with open(model, 'r') as f:
                try:
                    predictions[model] = json.load(f)  # Load the predictions for the model
                except json.JSONDecodeError:
                    print(f"Error: {model} is not a valid JSON file.")
                    continue
    return predictions


if __name__ == '__main__':
    parser = ArgumentParser(description="Choose best QA agents for each question across internal and external scenes for the AI City Challenge, Track 2, 2025.")
    parser.add_argument("--models", type=str, help="comma separated list of paths to predictions files based on the chosen models", required=True)
    parser.add_argument("--gt_dirs", nargs="+", required=True, help="List of root dirs for GT: e.g., wts_vqa/val bdd_vqa/val")
    parser.add_argument("--questions_file", type=str, default="WTS_VQA_PUBLIC_TEST.json", help="directory to save the results")
    parser.add_argument("--output_path", type=str, default="multi_agent_results.json", help="file to save the results in")
    parser.add_argument("--best_agents_path", type=str, default="best_qa_agents.json", help="file to save the best models in")

    args = parser.parse_args()

    # Read predictions
    predictions = load_prediction_models(args.models)
    gt = load_gt_from_val_dirs(args.gt_dirs)
    # Gather questions
    questions = augment_gt_with_internal_flag(gt)
    # Choose best models
    answers, best_models = choose_best_models(predictions, questions)
    # Save the answers
    with open(args.output_path, 'w') as f:
        json.dump(answers, f, indent=2)
    # store best models
    bms = []
    for k, v in best_models.items():
        question, internal = k.split('_')
        bms.append({
            'question': question,
            'internal': internal == 'True',
            'model': v[0],
            'accuracy': v[1]
        })
    with open(args.best_agents_path, 'w') as f:
        json.dump(sorted(bms, key=lambda x: x['accuracy'], reverse=True), f, indent=2)
    # Compute and print accuracy of final answers
    accuracy, question_accuracy = qa_accuracy(answers, gt, compute_question_accuracies=True)
    print(f"Overall accuracy: {accuracy:.2f}%")
    # print question accuracies in sorted order by decreasing accuracy value
    for question in sorted(question_accuracy.keys(), key=lambda x: question_accuracy[x]['accuracy'], reverse=False):
        stats = question_accuracy[question]
        print(f"  {question}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
    print("")
