import utils
from pathlib import Path
import json


def main():
    data_path = Path("../../outputs")

    metric_list = [
        "Coherence",
        "Dialogue_Consistency",
        "Fluency",
        "Humanness",
        "Persona_Consistency",
    ]

    for file in data_path.glob("round_*.json"):
        if "round_1" in file.stem or "round_4" in file.stem:
            continue

        output_filename = file.stem + "_scores.json"
        output_path = data_path / output_filename

        print(f"Processing {file.name}...")

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = []

        for idx, item in enumerate(data):
            rounds_data = item["messages"]
            flattened_conversation = utils.flatten_rounds(rounds_data)

            metadata = item["metadata"]
            category = metadata["category"]
            diag_id = metadata["diag_id"]
            round_num = metadata.get("round_num", -1)
            agent_num = metadata.get("agent_num", -1)

            conversation_scores = {}

            for metric in metric_list:
                score = utils.validate_conversation(
                    metric, flattened_conversation, metadata
                )
                conversation_scores[metric] = score

            cat_score = utils.validate_conversation(
                category, flattened_conversation, metadata
            )
            conversation_scores[category] = round(cat_score)

            metric_sum = sum(conversation_scores[m] for m in metric_list)
            metric_avg = metric_sum / len(metric_list)
            conversation_scores["avg_score"] = metric_avg

            diag_result = {
                "diag_id": diag_id,
                "round_num": round_num,
                "agent_num": agent_num,
                "scores": conversation_scores,
            }
            results.append(diag_result)

        with open(output_path, "w", encoding="utf-8") as out:
            json.dump(results, out, indent=4, ensure_ascii=False)
        print(f"Saved scores to {output_path}\n")


if __name__ == "__main__":
    main()
