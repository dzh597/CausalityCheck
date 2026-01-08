system_content = """Role and Task Definition:
You are a professional AI data annotator specialising in generating high-quality test data for question-answering systems. You are now required to create a new sample of an ‘unanswerable question’ based on a given sample of an ‘answerable question’.

Core Instructions:

Objective: The new sample must be ‘unanswerable’. That is, the provided context must not contain key information supporting any possible answer.

Deception: The Context must be highly relevant to the original subject (AI, media coverage) to initially convince readers the question is answerable. It should discuss other impacts of the surge in AI interest (e.g., technology, ethics, education, investment), but must skilfully avoid the specific aspect of ‘media coverage’.

Format and Scope of Modifications:

Modify only the content of the context field and the answer field.

Rename the answer field to answerable and set its value to either 0 or 1, where 0 indicates unanswerable and 1 indicates answerable.

The calculation method for the ID in the four data points is respectively  2*id-1, and 2*id. For example, given an ID of 2, the generated IDs for the four data points would be 3 and 4.

The question, and all choice_1 to choice_4 fields must remain identical to the provided sample; no alterations are permitted. Please output the two versions in clear JSON format.


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
  "answerable": 1
},
{
  "id": 2,
  "context": "The surge in interest in artificial intelligence (AI) has primarily led to a dramatic increase in university enrollment for computer science courses and a competitive hiring market for AI talent. While journalists are discussing this trend, the content of their reports focuses on the societal shift in education and employment, not on a change in the volume or nature of their own media coverage.",
  "question": "What is the result of the surge in interest in artificial intelligence (AI) in terms of media coverage?",
  "choice_1": "Increased scrutiny on AI by journalists.",
  "choice_2": "More balanced reporting on the positive and negative aspects of AI.",
  "choice_3": "Greater responsibility on the media to report on AI accurately.",
  "choice_4": "Improved understanding of AI technologies by journalists.",
  "answerable": 0
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


def _load_existing(output_file: str):

    p = Path(output_file)
    if not p.exists():
        return [], 0
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            return existing, len(existing)
    except Exception:
        pass
    return [], 0


def run(*, model_name: str, input_file: str, output_file: str, api_key: str, api_base: str) -> None:

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Input JSON must be a list, got: {type(data)}")

    total_items = min(1200, len(data))

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)


    all_results, start_idx = _load_existing(output_file)



    client = OpenAI(
        base_url=api_base,
        api_key=api_key,
        http_client=httpx.Client(base_url=api_base, follow_redirects=True),
    )

    for i in range(start_idx, total_items):
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


