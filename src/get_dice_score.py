import math
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
client = OpenAI()


def load_tool_docs(tool_docs_path):
    with open(tool_docs_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    func_map = {}
    for fdoc in data.get("functions", []):
        fname = fdoc["function"]
        func_map[fname] = {
            "desc": fdoc.get("desc", ""),
            "parameters": fdoc.get("parameters", []),
            "return": fdoc.get("return", {}),
        }
    return func_map


def find_function_doc(fn_name, func_map):
    if fn_name in func_map:
        return func_map[fn_name]

    fn_name_clean = fn_name.lower().replace("_", "").replace("-", "")
    for key in func_map.keys():
        key_clean = key.lower().replace("_", "").replace("-", "")
        if fn_name_clean == key_clean:
            return func_map[key]
    return None


def count_items_in_utterance_gpt(utterance_text, func_docs):
    if not func_docs:
        return 0

    docs_str = json.dumps(func_docs, indent=2, ensure_ascii=False)
    prompt = (
        "Below is documentation for one or more functions (description, parameters).\n"
        "Given the user utterance, count how many items (function names, parameter are semantically referenced in that utterance.\n\n"
        f"Function Docs:\n{docs_str}\n\n"
        f'User Utterance: "{utterance_text}"\n\n'
        "Respond with an integer only (0 if nothing is referenced)."
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=64,
        )
        num = int(completion.choices[0].message.content.strip())
        return num
    except:
        return 0


ALPHA_DEFAULT = math.e**2


def dice_formula(
    s_vector: list[int], total_items: int, alpha: float = ALPHA_DEFAULT
) -> float:
    if total_items == 0 or not s_vector:
        return 0.0

    num_utterances = len(s_vector)
    num_nonzero = sum(1 for x in s_vector if x > 0)

    denom = sum(math.log(1 + alpha * x) for x in s_vector if x > 0)
    if denom == 0:
        return 0.0

    numer = min(num_nonzero, total_items) * math.sqrt(num_utterances * total_items)
    return numer / denom


def compute_dice_score_for_dialogue(dialogue, func_map, alpha: float = ALPHA_DEFAULT):
    rounds = dialogue.get("rounds", [])
    messages = dialogue.get("messages", [])
    for r in rounds:
        break
    T = sum(1 + len(r.get("parameters", {})) for r in rounds)
    n_rounds = len(rounds)

    forward_docs_by_round = {}
    for i in range(1, n_rounds + 1):
        docs = []
        for j in range(i, n_rounds + 1):
            fn_doc = find_function_doc(rounds[j - 1]["function"], func_map)
            if fn_doc:
                docs.append(
                    {
                        "function": rounds[j - 1]["function"],
                        "desc": fn_doc["desc"],
                        "parameters": fn_doc["parameters"],
                        "return": fn_doc["return"],
                    }
                )
        forward_docs_by_round[i] = docs

    utterances = []
    for block in messages:
        for round_key, msgs in block.items():
            r = int(round_key.replace("Round", "").strip())
            for msg in msgs:
                if msg.get("speaker") != "AI Assistant":
                    utterances.append((r, msg.get("message", "")))

    matched_counts = [
        count_items_in_utterance_gpt(text, forward_docs_by_round.get(r, []))
        for r, text in utterances
    ]

    return dice_formula(matched_counts, T, alpha)


def compute_all_dice_scores(dataset, func_map, alpha):
    results = []
    for i, dialogue in enumerate(dataset, start=1):
        diag_id = dialogue.get("diag_id", i)
        dice = compute_dice_score_for_dialogue(dialogue, func_map, alpha)
        results.append((diag_id, dice))
    return results


def main():
    start_time = time.time()

    from pathlib import Path
    tool_docs_path = str(Path(__file__).resolve().parent / "graph" / "tool_docs.json")
    func_map = load_tool_docs(tool_docs_path)
    alpha = ALPHA_DEFAULT
    for i in range(3, 5):
        with open(f"./data/round_{i}.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)

        dice_scores = compute_all_dice_scores(dataset, func_map, alpha)

        if not dice_scores:
            print("Empty dataset.")
            return

        total = sum(score for _, score in dice_scores)
        avg = total / len(dice_scores)

        print("\n==== DICE Scores ====")
        for diag_id, score in dice_scores:
            print(f"Dialogue {diag_id}: DICE = {score:.4f}")
        print(f"\nAverage DICE Score: {avg:.4f}")
        print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
