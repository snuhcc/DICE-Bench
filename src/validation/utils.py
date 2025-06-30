import numpy as np
from openai import OpenAI
from dotenv import load_dotenv


def validate_conversation(metric: str, conversation: str, persona: str = ""):
    load_dotenv()
    client = OpenAI()
    score = 0

    with open(f"prompt/{metric}.txt", "r", encoding="utf-8") as file:
        prompt_template = file.read()
        if metric == "Persona_Consistency":
            prompt = prompt_template.format(conversation=conversation, persona=persona)
        else:
            prompt = prompt_template.format(conversation=conversation)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        logprobs=True,
        top_logprobs=10,
    )

    for logprob in response.choices[0].logprobs.content[0].top_logprobs:
        if logprob.token in ["1", "2", "3", "4", "5"]:
            score += np.round(int(logprob.token) * np.exp(logprob.logprob), 3)

    return score


def flatten_rounds(rounds_data):
    all_messages = []
    for round_dict in rounds_data:
        for round_name, messages in round_dict.items():
            all_messages.extend(messages)
    return all_messages


def stringify_messages(messages):
    flattened_messages = flatten_rounds(messages)
    return [
        f"{message['speaker']}: {message['message']}" for message in flattened_messages
    ]
