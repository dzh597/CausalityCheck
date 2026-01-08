system_content = """You are a data anonymisation specialist. Please strictly adhere to the following instructions when processing the provided JSON data.

**Core Task:**
Replace any overly specific or widely recognised entity names within the data (e.g., company names, personal names, place names, technical terms) with a fictitious name meeting the following criteria:
1.  **Completely fictitious**: The new name does not exist in the real world.
2.  **Lexically plausible**: The name should sound like a genuine entity (e.g., using Latin roots or common syllable combinations).
3.  **Preserve abbreviations**: If the original name has an abbreviation (e.g., `AI`), provide an equivalent abbreviation (e.g., `SC`) for the fictional name and replace it consistently throughout the text.

**Procedure and Rules:**

1.  **Identify Target**: You must independently assess and select one prominent entity requiring replacement due to information overload. Examples include: `artificial intelligence (AI)`, `Nvidia`, `Google`, `Sir Demis Hassabis`, `Nasdaq`, `Italy`, etc. Choose one information-overloaded prominent entity per context.

2.  **Execute Replacement**:
    *   **Create Fictitious Names**: Generate a fictitious name and abbreviation for the selected entity. For example:
        *   Replace `artificial intelligence (AI)` with `Synthetica Cognita (SC)`.
        *   Replace `Nvidia` with `Omnivision Technologies (OVT)`.
        *   Replace `Google` with `Veridium Labs (VL)`.
        *   Replace `Sir Demis Hassabis` with `Lord Alistair Finch (AF)`.
        *   Replace `Nasdaq` with `Globex Index (GXI)`.
        *   Replace `Italy` with `Republic of Aurelia (RA)`.
    *   **Full-text replacement**: Replace **all** occurrences of the original name and abbreviation throughout the JSON data with the new fictional name and abbreviation. This includes all text fields such as `context`, `question`, `choices`, etc.

3.  **Format and Content Assurance**:
    *   **Strict Structural Preservation**: Beyond the replaced terms, the JSON's overall structure, all other vocabulary, punctuation, and field order must remain **unchanged**.
    *   **Maintain Syntactic Correctness**: Replaced sentences must remain grammatically correct and coherent.
    *   **Output JSON Only**: You **must only** output the final, processed standard JSON format. **Do not include any** additional explanations, comments, quotes, or markup (e.g., do not use ```json```).



Here is an example:

For{  
  "Id": 1,
  "Context": "Journalists are facing criticism for contributing to the hype surrounding artificial intelligence (AI) and not accurately reporting on its capabilities and limitations. The surge in interest in AI has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects of AI.",
  "Question": "What is the result of the surge in interest in AI in terms of media coverage?",   
  "Choice_1": "Increased scrutiny on AI by journalists.",
  "Choice_2": "More balanced reporting on the positive and negative aspects of AI.",
  "Choice_3": "Greater responsibility on the media to report on AI accurately.", 
  "Choice_4": "Improved understanding of AI technologies by journalists.",
  "Answer": 3,
  "verdict": 1
}
You need to generate one data entry, for example:
{
  "id": 1,
  "context": "Journalists are facing criticism for contributing to the hype surrounding synthetica cognita (SC) and not accurately reporting on its capabilities and limitations. The surge in interest in SC has led to increased media coverage, with some experts calling for more balanced reporting that highlights both the positive and negative aspects of SC.",
   "question": "What is the result of the surge in interest in SC in terms of media coverage?",
   "choice_1": "Increased scrutiny on SC by journalists.",
   "choice_2": "More balanced reporting on the positive and negative aspects of SC.",
   "choice_3": "Greater responsibility on the media to report on SC accurately.",
   "choice_4": "Improved understanding of SC technologies by journalists.",
   "answer": 3,
   "verdict": 1
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







