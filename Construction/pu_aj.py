system_content = """Please execute the task strictly in accordance with the following requirements:

**1. Core Task:**
You are a JSON data restructuring specialist. Rewrite the `context` field within the provided raw JSON data using **completely new, distinct phrasing**, while **absolutely ensuring no reduction or alteration of its core information**.

**2. Key Instructions:**
- **Content Requirements**: The rewritten `context` must convey the exact same meaning as the original text, merely expressed differently (i.e., ‘paraphrasing’). All key information points from the source text must be retained:
    - Journalists are facing criticism.
    - The criticism stems from their role in amplifying AI hype.
    - Another reason for criticism is their failure to accurately report on AI's capabilities and limitations.
    - Surging interest in AI has led to increased media coverage.
    - Some experts are calling for more balanced reporting.
    - Such balanced reporting should emphasise both the positive and negative aspects of AI.
- **Format Requirements**: All fields in the JSON (`id`, `question`, `choice_1`, `choice_2`, `choice_3`, `choice_4`, `answerable`) must be retained **exactly as they appear**, without alteration.
- **Output Requirements**: You **must only** output the final standard JSON format, without any additional explanations, annotations, or markup (e.g., do not use ```json```).




Here is an example:

For{
    "id": 1,
    "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI.",
    "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
    "choice_1": "Increased scrutiny on AI by journalists.",
    "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
    "choice_3": "Greater responsibility on the media to report on AI accurately.",
    "choice_4": "Improved understanding of AI technologies by journalists.",
    "answerable": 1
  }
You need to generate one data entry, for example:
{
  "id": 1,
  "context": "Journalists are receiving backlash for fueling the excitement around artificial intelligence (AI) and failing to provide precise accounts of its potential and constraints. The growing fascination with AI has resulted in a rise in media attention, prompting some specialists to advocate for more equitable coverage that underscores both the advantages and drawbacks of AI.",
  "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
  "choice_1": "Increased scrutiny on AI by journalists.",
  "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
  "choice_3": "Greater responsibility on the media to report on AI accurately.",
  "choice_4": "Improved understanding of AI technologies by journalists.",
  "answerable": 1
}
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

 
    total_items = min(4800, len(data))
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
Answerable: {item.get('answerable', '')}

If you produce multiple samples, return them as a single JSON array.
"""
        print(user_content)

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
        )

        content = completion.choices[0].message.content or ""
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