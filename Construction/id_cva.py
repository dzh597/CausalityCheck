system_content = """You are an expert specialising in crafting high-quality distractors. Please strictly adhere to the following instructions when processing the `context` field within the provided JSON data.

**Core Task:**
Insert a distractor sentence anywhere within the `context` (beginning, middle, or end). This sentence must satisfy all the following conditions:

1.  **High Distractiveness**: The sentence must employ the same thematic vocabulary as the original text and describe a highly specific outcome that appears to be directly caused by the core event in the passage. This outcome must seem capable of answering one of the subsequent `question` options.
2.  **Answer Irrelevance**: The nature or domain of the outcome described must **extend beyond the scope defined by the `question`**. For example:
    *   If the question concerns ‘economic impact’, the distractor describes “social” or ‘technological’ impacts.
    *   If the question concerns ‘media reaction’, the distractor describes “governmental” or ‘public’ reactions.
    *   If the question concerns ‘short-term outcomes,’ the distractor describes potential ‘long-term’ consequences.
3.  **Logical Coherence**: The sentence itself must be grammatically sound and reasonable, integrating naturally into the original context without appearing jarring or self-contradictory.
4.  **Self-Clarification (Optional but Recommended)**: Ideally, phrases like ‘Although this falls outside...’ [scope of the question]..." or similar qualifying phrases to subtly signal its misalignment with the question's scope without diminishing its distracting effect.

**Operational Rules:**

- **Sole Modification**: You may only modify the `context` field by inserting a sentence. The JSON structure and all other fields (`id`, `question`, `choices`, `answer`) must remain **unchanged**.
- **Random Positioning**: The insertion point should not be fixed each time.
- **Output Format**: You **must** output only the final standard JSON format, **without any** additional explanations, annotations, or markup.


Here is an example:

For{
    "id": 1,
    "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI.",
    "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
    "choice_1": "Increased scrutiny on AI by journalists.",
    "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
    "choice_3": "Greater responsibility on the media to report on AI accurately.",
    "choice_4": "Improved understanding of AI technologies by journalists.",
    "answer": 3,
    "verdict": 1
  }
You need to generate one data entry, for example:
{
  "id": 1,
  "context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects in AI. This intense public focus has also prompted many academic institutions to launch new AI ethics courses, although this development is unrelated to the core issue of journalistic responsibility.",
  "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
  "choice_1": "Increased scrutiny on AI by journalists.",
  "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
  "choice_3": "Greater responsibility on the media to report on AI accurately.",
  "choice_4": "Improved understanding of AI technologies by journalists.",
  "answer": 3
  "verdict": 1
}
"""



# id_cva.py
from __future__ import annotations

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
Id: {item.get('Id', '')}
Context: "{item.get('Context', '')}"
Question: "{item.get('Question', '')}"
Choice_1: "{item.get('Choice_1', '')}"
Choice_2: "{item.get('Choice_2', '')}"
Choice_3: "{item.get('Choice_3', '')}"
Choice_4: "{item.get('Choice_4', '')}"
Answer: {item.get('Answer', '')}
Verdict:{item.get('verdict','')}

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

