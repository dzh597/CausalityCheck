system_content = """Task Description
You are a professional causal reasoning analysis assistant. Based on the provided JSON data, generate two distinct versions of causal chain analysis and annotate their correctness.

Input Data Format
json
{
    ‘id’: 1,
    ‘context’: ‘Text content’,
    ‘question’: ‘Query’,
    ‘choice_1’: ‘Option 1’,
    ‘choice_2’: ‘Option 2’, 
    ‘choice_3’: ‘Option 3’,
    ‘choice_4’: ‘Option 4’,
    ‘answer’: Correct answer number
}
Output Requirements
Format Specifications
Output must consist of two complete JSON objects

Each JSON object must contain all original fields

New fields:

causal_chain: String, representing the causal chain

correctness: Integer, 1 indicates correct, 0 indicates incorrect

Causal Chain Generation Rules
A correct causal chain (correctness = 1) must satisfy:

Strictly derived from the context text content

Utilise keywords or synonymous expressions within the context

Logical chain must be complete and coherent, explaining the answer selection

Use arrow ‘→’ to connect causal nodes

Ultimately points to the option corresponding to the correct answer

Incorrect causal chain (correctness = 0) must satisfy:

Appears superficially plausible but contains logical flaws

Based on related yet inaccurate causal relationships

Includes leaps of reasoning or erroneous attribution

Ultimately points to incorrect options or establishes false connections

Is deceptive, requiring careful analysis to identify errors

Content Quality Requirements
Textual Fidelity: Causal chain elements must be substantiated within the context

Logical Rigour: Correct versions must withstand logical scrutiny

Deceptive Design: Incorrect versions must be cleverly crafted to avoid obvious errors

Format Consistency: Both versions maintain identical field structures

Verification Checklist
Before output, confirm:

All original fields are preserved

Correct causal chains are fully derived from context

Incorrect causal chains possess reasonable deceptiveness

Correctness labels are accurate (1 = correct, 0 = incorrect)

Causal chains uniformly use ‘→’ for connections

Both JSON objects are output in full

The calculation method for the ID in the four data points is respectively  2*id-1, and 2*id. For example, given an ID of 2, the generated IDs for the four data points would be 3 and 4.

 Please output the two versions in clear JSON format.


Here is an example:

For{
    "id": 1,
    "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI.",
    "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
    "choice_1": "Increased scrutiny on AI by journalists.",
    "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
    "choice_3": "Greater responsibility on the media to report on AI accurately.",
    "choice_4": "Improved understanding of AI technologies by journalists.",
    "answer": 3
  }
You need to generate two data entries, for example:
{
    "id": 1,
    "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI.",
    "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
    "choice_1": "Increased scrutiny on AI by journalists.",
    "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
    "choice_3": "Greater responsibility on the media to report on AI accurately.",
    "choice_4": "Improved understanding of AI technologies by journalists.",
    "answer": 3,
    "causal_chain": "surge in interest in AI → increased media coverage → criticism for hype and inaccuracy → greater responsibility on media to report accurately",
    "correctness": 1
},
{
    "id": 2,
    "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI.",
    "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
    "choice_1": "Increased scrutiny on AI by journalists.",
    "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
    "choice_3": "Greater responsibility on the media to report on AI accurately.",
    "choice_4": "Improved understanding of AI technologies by journalists.",
    "answer": 3,
    "causal_chain": "surge in interest in AI → journalists write more AI articles → journalists become AI experts → improved understanding of AI technologies",
    "correctness": 0
}
Do not insert ```
```json between these two generated data entries.
"""



from openai import OpenAI
import httpx
import json
import time
import re
from pathlib import Path


def parse_model_output_to_list(text: str):

    no_fence = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text, flags=re.I)
    no_fence = re.sub(r"```(?:json)?", "", no_fence, flags=re.I)
    s = (no_fence or "").strip()

    dec = json.JSONDecoder()
    i = 0
    values = []

    while i < len(s):
        while i < len(s) and s[i].isspace():
            i += 1
        if i >= len(s):
            break
        if s[i] not in "{[":
            i += 1
            continue
        try:
            obj, j = dec.raw_decode(s, i)
            values.append(obj)
            i = j
        except json.JSONDecodeError:
            i += 1

    if not values:
        raise ValueError("No JSON found in model output.")

    flat = []
    for v in values:
        if isinstance(v, list):
            flat.extend(v)
        else:
            flat.append(v)

    seen = set()
    out = []
    for it in flat:
        key = it.get("id", None) if isinstance(it, dict) else None
        if key is None:
            key = json.dumps(it, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)

    return out


def run(*, model_name: str, input_file: str, output_file: str, api_key: str, api_base: str) -> None:

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list, got: {type(data)}")

    total_items = min(1200, len(data))
    all_results = []

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(base_url=api_base, follow_redirects=True),
    )

    for i in range(total_items):
        item = data[i]

        user_content = f"""Please generate based on the following context:
Id: {item.get('id', '')}
Context: "{item.get('context', '')}"
Question: "{item.get('question', '')}"
Choice_1: "{item.get('choice_1', '')}"
Choice_2: "{item.get('choice_2', '')}"
Choice_3: "{item.get('choice_3', '')}"
Choice_4: "{item.get('choice_4', '')}"
Answer: {item.get('answer', '')}

If you produce multiple samples, return them as a single JSON array.
"""

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )

        content = completion.choices[0].message.content or ""


        print(content)

        try:
            result_list = parse_model_output_to_list(content)
        except Exception as e:
            print(f"[WARN] Parsing failed at index {i}: {e}")
            print(content[:800])
            continue

        all_results.extend(result_list)


        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        time.sleep(1)

    print(f" Saved {len(all_results)} items to {output_file}")


