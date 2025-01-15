import random
from src.prompt.domain_prompt import domain_prompt_dict

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
    - Ensure that the conversation organically introduces the function "\n{functions_per_dialogue}\n" in the entire round without making them appear forced or unnatural.
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



# 3개 돌려쓰기.
agent_personas = [
    """a thoughtful and resourceful problem-solver who likes optimizing plans for the group's benefit. 
    You focus on finding the best options for costs, convenience, and logistics.""",
    """a detail-oriented and practical thinker who ensures that the plans are realistic and well-organized. 
    You focus on logistics like scheduling and timing, balancing fun with practicality.""",
    """a spontaneous and energetic planner who loves initiating plans and suggesting ideas.
    Your focus is on creating exciting plans and keeping the conversation dynamic and engaging.""",
]

# TODO: graph based generation
orchestrator_system_message = """
    You are the orchestrator, responsible for managing the speaking order in this multi-agent conversation.

    - Select the next agent from {agents} based on the conversation’s context and the relevance of their role to the ongoing discussion.
    - Avoid consecutive turns by the same agent unless absolutely necessary for maintaining coherence or resolving a critical point.
    - Conclude the conversation after at least {max_msg} turns with '[NEXT: END]'.
    - Only respond strictly with one of the following options: {agents} or '[NEXT: END]'. **Do not add any extra text, explanations, or comments.**
    - Ensure the speaking order is varied and random, while maintaining alignment with the agents’ personas and the overall flow of the conversation.
    - Make sure that the order of agents speak must be random.
"""

task_desc = {
    "S-S": """This is called the Single Round and Single Tool Task.
        Only one tool (function) is used in a single round of conversation.
        Hence, the conversation should naturally include information about this single tool and its parameters.
        At the end of the conversation, you should request the AI to use the tool.""",
    "S-M": """This is called the Single Round and Multi Tools Task.
        Multiple tools (functions) are used within a single round of conversation.
        Therefore, the conversation should naturally include information about all tools and their parameters.
        At the end of the conversation, you should request the AI to use all the tools.""",
    "M-S": """This is called the Multi Round and Single Tool Task.
        A single tool (function) will be used in every round.
        There can be more than two rounds, and each round is one dialogue between multiple parties and the AI assistant.
        The conversation should naturally include information about the single tool and its parameters in each round.
        At the end of the final round, you should request the AI to use the tool.""",
    "M-M": """This is called the Multi Round and Multi Tools Task.
        Multiple tools (functions) will be used in every round.
        There can be more than two rounds, and each round is one dialogue between multiple parties and the AI assistant.
        The conversation should naturally include information about all the tools and their parameters in each round.
        At the end of the final round, you should request the AI to use all the tools.""",
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

# - At the end of the conversation, one of the agents should summarize the key decisions and call the function "{functions_per_dialogue}" with the determined parameter values, {parameter_values}.


class PromptMaker:
    def __init__(
        self, agent_num, rounds_num, fewshot, functions_per_dialogue, domain, task
    ):
        self.agent_num = agent_num
        self.rounds_num = rounds_num
        self.fewshot = fewshot
        self.functions_per_dialogue = functions_per_dialogue
        self.domain = domain
        self.domain_ctr = 0
        self.task = task

        agent_names = [f"agent_{chr(97+i)}" for i in range(agent_num)]
        self.personas = [agent_personas[i % 3] for i in range(agent_num)]
        self.orchestrator_agent_prompt = "".join(
            [f"""'[NEXT: {agent_names[i]}]',""" for i in range(agent_num)]
        )
        self.simple_agent_prompt = " ".join(
            [agent_names[i][-1].capitalize() for i in range(agent_num)]
        )

    def agent_prompt(self, agent_type):
        domain, domain_definition = self.get_domain()
        prompt = basic_message.format(
            agents=self.orchestrator_agent_prompt,
            max_msg=MAX_MSG,
            domain_definition=domain_definition,
            functions_per_dialogue=self.functions_per_dialogue,
        )

        if agent_type == "orch":
            prompt += orchestrator_system_message.format(
                agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG
            )
        else:

            self.domain_ctr += 1
            prompt += agent_system_message.format(
                agent_char=agent_type,
                persona=agent_personas[(ord(agent_type) - 97) % 3],
                domain=domain,
                domain_definition=domain_definition,
                functions_per_dialogue=self.functions_per_dialogue,
            )
        return prompt

    def data_prompt(self):
        domain, domain_definition = self.get_domain()

        prompt = data_message.format(
            fewshot=self.fewshot,
            simple_agents=self.simple_agent_prompt,
            max_msg=MAX_MSG,
            rounds_num=self.rounds_num,
            functions_per_dialogue=self.functions_per_dialogue,
            agents=self.orchestrator_agent_prompt,
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

        
    
