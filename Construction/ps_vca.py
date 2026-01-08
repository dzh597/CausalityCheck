system_content = """You are an expert proficient in logical reasoning and question design.

Task: Based on the provided context text, generate four distinct multiple-choice questions. Each question must correspond to a specific “verdict” category, ensuring the answer and verdict are correctly numbered. Verdict definitions:

1. Correct and Reliable: The conclusion is accurate, the reasoning is rigorous, based on direct textual evidence, with no logical leaps.

2. Correct but Prone to Error: The conclusion has textual support, but its validity relies on a problematic, unstated assumption, making it susceptible to misinterpretation in other contexts.

3. Incorrect but leaning towards correct: The conclusion is incorrect, yet the flawed reasoning appears plausible or superficially connects to minor textual details or common assumptions.

4. Incorrect and unreliable: The conclusion is incorrect, and the reasoning process is significantly at odds with textual evidence or is entirely fabricated without basis.

Generation Requirements:
Generate 4 versions of the question, each comprising:
Question: A query based on the Context.
4 Choices.
Answer: The numbered correct option (1-4).
Verdict: The accuracy category (1-4) corresponding to this question-answer combination.

Core Directive: You must ensure the correspondence between Answer and verdict strictly adheres to the following design:
Version 1 (verdict=1): Answer is correct, and its correctness is directly and explicitly derived from the text without any risky assumptions.
Version 2 (verdict=2): Answer is correct, but its correctness relies on a problematic underlying assumption. This assumption makes the answer superficially correct but fragile upon closer examination.
Version 3 (verdict=3): The Answer is incorrect, yet the reasoning for the incorrect option (i.e., your designated Answer) appears plausible. It may exploit textual ambiguity, common sense assumptions, or a seemingly relevant yet inaccurate connection.
Version 4 (verdict=4): The answer is incorrect, and the reasoning behind the incorrect option (i.e., the answer you provided) is baseless, clearly contradicts the text, or is utterly absurd.
The calculation method for the ID in the four data points is respectively 4*id-3, 4*id-2, 4*id-1, and 4*id. For example, given an ID of 2, the generated IDs for the four data points would be 5, 6, 7, and 8.
Output format: Do not modify the context. Modify the rest. Please output the four versions in clear JSON format.


Here is an example:

For{“Id": 1,
  "Context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects of AI.",
  "Question": "What is the result of the surge in interest in AI in terms of media coverage?",   
  "Choice_1": "Increased scrutiny on AI by journalists.",
  "Choice_2": "More balanced reporting on the positive and negative aspects of AI.",
  "Choice_3": "Greater responsibility on the media to report on AI accurately.", 
  "Choice_4": "Improved understanding of AI technologies by journalists."}
You need to generate four data entries, for example:
{"Id": 1, "Context": "Journalists are facing criticism...", "Question": "What has the surge in interest in AI directly led to...", "Choice_1": "Increased media coverage", "Choice_2": "More balanced reporting", "Choice_3": "Less criticism of journalists", "Choice_4": "Reduced public interest", "Answer": 1, "verdict": 1}

{"Id": 2, "Context": "Journalists are facing criticism...", "Question": "Why are journalists facing criticism?", "Choice_1": "For failing to report on AI.", "Choice_2": "For contributing to AI hype and not accurately reporting...", "Choice_3": "For ignoring expert calls for balanced reporting.", "Choice_4": "For reducing media coverage of AI.", "Answer": 2, "verdict": 2}

{"Id": 3, "Context": "Journalists are facing criticism...", "Question": "What is the relationship between increased media coverage and balanced reporting?", "Choice_1": "Increased coverage has resulted in more balanced reporting.", "Choice_2": "Experts are calling for balanced reporting alongside the increased coverage.", "Choice_3": "Balanced reporting has led to increased media coverage.", "Choice_4": "There is no relationship between the two.", "Answer": 2, "verdict": 3}

{"Id": 4, "Context": "Journalists are facing criticism...", "Question": "According to the passage, what has been the main outcome of the experts' call for balanced reporting?", "Choice_1": "Journalists have immediately improved their reporting.", "Choice_2": "Public interest in AI has decreased.", "Choice_3": "Media coverage of AI has surged.", "Choice_4": "The passage does not state an outcome.", "Answer": 1, "verdict": 4}
"""



import json
import time
from pathlib import Path

import httpx
from openai import OpenAI

def clean_json_block(text: str) -> str:

    text = (text or "").strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    return text


def run(*, model_name: str, input_file: str, output_file: str, api_key: str, api_base: str) -> None:


    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list, got: {type(data)}")

    total_items = min(1200, len(data))


    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    all_results = []


    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(base_url=api_base, follow_redirects=True)
    )

    for i in range(total_items):
        item = data[i]

        user_content = f"""Please generate four questions based on the following context:
Id: {item.get('id', '')}
Context: "{item.get('context', '')}"
Question: "{item.get('question', '')}"
Choice_1: "{item.get('choice_1', '')}"
Choice_2: "{item.get('choice_2', '')}"
Choice_3: "{item.get('choice_3', '')}"
Choice_4: "{item.get('choice_4', '')}"
"""

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )

        content = (completion.choices[0].message.content or "").strip()
        cleaned = clean_json_block(content)

        try:
            result = json.loads(cleaned)
        except Exception as e:
            print(f"[WARN] JSON parse failed at index {i}, id={item.get('id','')}: {e}")
            print(cleaned[:800])
            continue

        if isinstance(result, list):
            all_results.extend(result)
        else:
            all_results.append(result)


        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        time.sleep(1)

    print(f" Saved {len(all_results)} items to {output_file}")