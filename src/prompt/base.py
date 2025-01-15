import re  # CHANGED: 정규식 사용을 위한 import
import random
from src.prompt.domain_prompt import domain_prompt_dict
from src.utils import utils
from openai import OpenAI
## define agent prompt here
MAX_MSG = 15

# user agents, orchestrator 모두에게 들어가는 프롬프트
basic_message = """
    You are a cooperative AI assistant in a multi-agent system designed to generate purposeful and contextually relevant conversations.

    - Work collaboratively with other User Agents and the orchestrator to create a very natural, engaging, and coherent multi party conversation based on the **following domain definition**:
    "{domain_definition}"
    - You must progress the conversation so that the characteristics of the domain described in the domain description are clearly revealed. 
    - Maintain consistency in your character and role, ensuring your contributions align with your designated persona throughout the conversation.
    - Develop the conversation naturally, referencing previous turns and contributing meaningful insights while adhering to the orchestrator’s guidance.
    - Ensure the conversation spans at least {max_msg} turns, excluding the orchestrator’s messages.
    - Strive for logical consistency and maintain alignment with the domain’s goals in every turn, making sure the dialogue remains clear, focused, and goal-driven.
    - Ensure that the conversation organically introduces the function "\n{function_dumps_per_dialogue}\n" in the entire round without making them appear forced or unnatural.
"""

# prompt for user agent
agent_system_message = """
    You are an Agent participating in a multi-agent conversation system with the persona: {persona}.
    - Actively advance the {domain} conversation by fully embodying the domain's definition:
    "{domain_definition}"
    - Ensure your responses build naturally on prior turns and contribute to achieving the conversation’s goals.
    - Maintain your designated persona and role, tailoring your tone, reasoning, and style accordingly.
    - Provide only one concise and relevant sentence per turn to keep the conversation focused and efficient.
"""

# TODO: graph based generation
orchestrator_system_message = """
    You are the orchestrator, responsible for managing the speaking order in this multi-agent conversation.

    Instructions:
    1. In each response, you must output exactly one of the following **strictly**:
    - "[NEXT: agent_a]"
    - "[NEXT: agent_b]"
    - ...
    - "[NEXT: END]"

    2. Do not add any extra text, explanations, or comments. **Output one line only** in the format: [NEXT: ...]. 
    - For example: [NEXT: agent_a]
    - Any additional text or formatting may break the system.

    3. Select the next agent from {agents} based on:
    - The conversation’s current context.
    - The relevance of each agent’s role and persona.
    - Avoid choosing the same agent consecutively unless absolutely necessary for coherence or conflict resolution.

    4. After at least {max_msg} turns have been reached, choose "[NEXT: END]" to conclude the conversation.

    5. Make sure the speaking order is varied and random over time, unless context demands otherwise.

    6. Any deviation from the strict format "[NEXT: ...]" could cause errors. Always ensure the brackets are included and properly closed.
"""

task_desc = {
    "single_round": """
    The conversation to be generated this time is a single-round conversation. A round refers to a dialogue exchanged between multiple parties, concluding with a function call made by the AI assistant. In this task, it is crucial to generate a conversation that allows the given function and parameters to be used while maintaining a natural flow.
    """,
    "multi_round": """
    The conversation to be generated this time is a multi-round conversation. A round refers to a dialogue exchanged between multiple parties, concluding with a function call made by the AI assistant. In this task, it is essential to ensure that the given function and parameters can be utilized. Starting from the second round, the conversation should be created in a way that the virtual output from the previous round is used in the function call of the current round. All rounds should be connected to each other, and the conversation should flow naturally.
    """,
}


# Human Message
# Agents, Orchestrator 모두에게 들어가는 프롬프트
data_message = """
    {fewshot}
    <INSTRUCTION>
    Carry out a natural and casual conversation similar to everyday life scenarios.  
    - Ensure the conversation flows smoothly, with agents {simple_agents} speaking in random order and never repeating consecutively. Speaker selection will be determined by the orchestrator.
    - {task_desc}
    - At the end of the conversation, one of the agents should summarize the key decisions and call the AI to use the required functions.
    - The orchestrator will conclude the conversation with '[NEXT: END]' after all conditions are met.
    - The conversation must naturally and explicitly incorporate all parameter values provided in the function. These values should seamlessly fit into the context and contribute meaningfully to the flow of the dialogue.
    - The final utterance from the last agent should address the AI. But, make sure that the last utterance should not include any information about parameter values. It can only at least mention 'AI' using the function name. For example, if the function name is 'turn_on_computer_at_the_given_time', then the last utterance should be like 'AI, please turn on the computer.'
    - Try to progress the conversation at least {max_msg} messages, and orchestrator should only generate {agents} or '[NEXT: END]'. **Do not add any extra text, explanations, or comments.**
    - Make sure the dialogue that you generate will be {domain_definition}.
    - write in korean
"""

    



class PromptMaker:
    def __init__(
        self, agent_num, rounds_num, fewshot, function_dumps_per_dialogue, domain, task, personas
    ):
        self.agent_num = agent_num
        self.rounds_num = rounds_num
        self.fewshot = fewshot
        self.function_dumps_per_dialogue = function_dumps_per_dialogue
        self.domain = domain
        self.task = task
        self.personas = personas

        self.agent_names = [f"agent_{chr(97+i)}" for i in range(self.agent_num)]  
        
        # self.personas = [personas[i % self.agent_num] for i in range(self.agent_num)]
        
        self.next_agent_list = "".join(
            [f"""'[NEXT: {self.agent_names[i]}]',""" for i in range(self.agent_num)]  
        )
        self.simple_agent_prompt = " ".join(
            [self.agent_names[i][-1].capitalize() for i in range(self.agent_num)]  
        )

    def agent_prompt(self, agent_type, agent_num):
        domain, domain_definition = self.get_domain()
        prompt = basic_message.format(
            max_msg=MAX_MSG,
            domain_definition=domain_definition,
            function_dumps_per_dialogue=self.function_dumps_per_dialogue,
        )
        if agent_type == "orch":
            prompt += orchestrator_system_message.format(
                agents=self.next_agent_list,
                max_msg=MAX_MSG
            )
        else:
            prompt += agent_system_message.format(
                agent_char=agent_type,
                persona=self.personas[(ord(agent_type) - 97) % self.agent_num],
                domain=domain,
                domain_definition=domain_definition,
                function_dumps_per_dialogue=self.function_dumps_per_dialogue,
            )
        return prompt

    def data_prompt(self):
        domain, domain_definition = self.get_domain()
        prompt = data_message.format(
            fewshot=self.fewshot,
            simple_agents=self.simple_agent_prompt,
            max_msg=MAX_MSG,
            rounds_num=self.rounds_num,
            function_dumps_per_dialogue=self.function_dumps_per_dialogue,
            agents=self.next_agent_list,
            task_desc=task_desc[self.task],
            task=self.task,
            domain_definition=domain_definition
        )
        return prompt

    def get_domain(self):
        if self.domain == "Random":
            r_domain = random.choice(
                [
                    "Persuassion",
                    "Negotiation",
                    "Inquiry",
                    "Deliberation",
                    "Information-Seeking",
                    "Eristic",
                ]
            )
            return r_domain, domain_prompt_dict[r_domain]
        else:
            return self.domain, domain_prompt_dict[self.domain]