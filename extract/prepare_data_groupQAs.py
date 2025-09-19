import json
import argparse

def extract_specific_questions(input_file_path, output_file_path):
    """
    Reads a .jsonl file, filters the 'conversations' to keep only specified
    questions, and writes the result to a new .jsonl file.

    Args:
        input_file_path (str): The path to the source .jsonl file.
        output_file_path (str): The path where the filtered .jsonl file will be saved.
    """
    # These are the questions you want to extract and keep.
    target_questions = {
        "What is the orientation of the pedestrian's body?",
        "What is the position of the pedestrian relative to the vehicle?",
        "What is relative distance of pedestrian from vehicle?",
        "What is the pedestrian's line of sight?",
        "What is the pedestrian's visual status?",
        "What is the pedestrian's direction of travel?",
        "What is the pedestrian's awareness regarding vehicle?",
        "What is the pedestrian's action?",
        "What is pedestrian's speed?",
        "What is the position of the vehicle relative to the pedestrian?",
        "What is relative distance of vehicle from pedestrian?",
        "What is vehicle's field of view?",
        "What is the action taken by vehicle?",
        "What is the fine-grained action taken by the pedestrian?",
    }

    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                try:
                   
                    data = json.loads(line)
                    conversations = data.get("conversations", [])
                    filtered_conversations = []

                  
                    for i in range(0, len(conversations), 2):
                        human_turn = conversations[i]
                       
                        if i + 1 < len(conversations):
                            gpt_turn = conversations[i+1]
                        else:
                            continue 

                        
                        if human_turn.get("from") == "human" and gpt_turn.get("from") == "gpt":
                            human_text = human_turn.get("value", "")
                            
                            # Extract the question from the human's text
                            current_question = ""
                            for text_line in human_text.split('\n'):
                                if '?' in text_line:
                                    current_question = text_line.strip()
                                    break
                            
                            # If the extracted question is in our target list, keep the pair
                            if current_question in target_questions:
                                filtered_conversations.append(human_turn)
                                filtered_conversations.append(gpt_turn)
                    
                    
                    if filtered_conversations:
                        data["conversations"] = filtered_conversations
                        
                        outfile.write(json.dumps(data) + '\n')

                except json.JSONDecodeError:
                    print(f"Warning: Skipping a line that is not valid JSON: {line.strip()}")
                except (KeyError, IndexError) as e:
                    print(f"Warning: Skipping a line due to unexpected format: {e} - Line: {line.strip()}")

        print(f"Processing complete. Filtered data saved to '{output_file_path}'")

    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Parses command-line arguments and calls the extraction function.
    """
    parser = argparse.ArgumentParser(
        description="Filters a .jsonl file to keep only specified questions and writes the output to a new file."
    )
    parser.add_argument(
        "input_file", 
        help="The path to the source .jsonl file."
    )
    parser.add_argument(
        "output_file", 
        help="The path where the filtered .jsonl file will be saved."
    )
    args = parser.parse_args()
    
    extract_specific_questions(args.input_file, args.output_file)

if __name__ == '__main__':
    main()

