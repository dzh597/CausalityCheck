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
    "answer": 3
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
}
"""



from openai import OpenAI
import httpx
import json
import time
from pathlib import Path




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


    total_items = min(2400, len(data))
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

        time.sleep(1)


        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

    print(f" Saved {len(all_results)} items to {output_file}")



