import random
from src.prompt.domain_prompt import domain_prompt_dict

## define agent prompt here
MAX_MSG = 15

# user agents, orchestrator 모두에게 들어가는 프롬프트
basic_message = """
    You are a cooperative AI assistant in a multi-agent system designed to generate purposeful and contextually relevant conversations.

    - Work collaboratively with other User Agents and the orchestrator to create a very natural, engaging, and coherent discussion based on the following domain definition:
    "{domain_definition}"
    - Maintain consistency in your character and role, ensuring your contributions align with your designated persona throughout the conversation.
    - Develop the conversation naturally, referencing previous turns and contributing meaningful insights while adhering to the orchestrator’s guidance.
    - Ensure the conversation spans at least {max_msg} turns, excluding the orchestrator’s messages.
    - Strive for logical consistency and maintain alignment with the domain’s goals in every turn, making sure the dialogue remains clear, focused, and goal-driven.
"""

# prompt for user agent
agent_system_message = """
    You are an Agent {agent_char} participating in a multi-agent conversation system.
    - Actively advance the {domain} conversation by fully embodying the domain's definition:
    "{domain_definition}"
    - Discuss and negotiate the following parameters as part of the conversation:
    {parameters}.
    - Ensure your responses build naturally on prior turns and contribute to achieving the conversation’s goals.
    - Maintain your designated persona and role, tailoring your tone, reasoning, and style accordingly.
    - Provide only one concise and relevant sentence per turn to keep the conversation focused and efficient.
    - Ensure that the conversation organically introduces the function "{func}" and its parameter values without making them appear forced or unnatural.
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
    'S-S': 'This task is called Single Round and Single Tool Task where only one tool is used in a single round. Therefore, the conversation should contain the information of a single tool and its parameters very naturally.',
    'S-M': 'This task is called Single Round and Multi Tools Task where multiple tools are used in a single round. Therefore, the conversation should contain the information of multiple tools and their parameters very naturally.' 
}


# function call output : GPT 생성 유도 (get_weather 같은거)
# topic: 좀더 구체화
data_message = """
    {fewshot}
    <INSTRUCTION>
    Carry out a natural and casual conversation similar to everyday life scenarios.  
    - Ensure the conversation flows smoothly, with agents {simple_agents} speaking in random order and never repeating consecutively. Speaker selection will be determined by the orchestrator.
    - {task_desc}
    - The conversation must naturally and explicitly incorporate all parameter values provided in {func}. These values should seamlessly fit into the context and contribute meaningfully to the flow of the dialogue.
    - At the end of the conversation, one of the agents should summarize the key decisions and call the function "{func}" with the determined parameter values, {parameter_values}.
    - The orchestrator will conclude the conversation with '[NEXT: END]' after all conditions are met.
    - The final utterance from the last agent should address the AI. But, make sure that the last utterance should not include any information about parameter values. It can only at least mention 'AI' using the function name. For example, if the function name is 'turn_on_computer_at_the_given_time', then the last utterance should be like 'AI, please turn on the computer.'
    - Try to progress the conversation at least {max_msg} messages, and orchestrator should only generate {agents} or '[NEXT: END]'. **Do not add any extra text, explanations, or comments.**
"""


class PromptMaker:
    def __init__(self, agent_num, round, fewshot, func, parameter_values, domain, iter, task):
        self.agent_num = agent_num
        self.round = round
        self.fewshot = fewshot
        self.func = func
        self.parameter_values = parameter_values
        self.domain = domain
        # self.now_domains = [self.get_domain() for i in range(iter)]
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
            agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG, domain_definition=domain_definition
        )

        if agent_type == "orch":
            prompt += orchestrator_system_message.format(
                agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG, round=self.round
            )
        else:
            
            self.domain_ctr += 1
            prompt += agent_system_message.format(
                agent_char=agent_type,
                persona=agent_personas[(ord(agent_type) - 97) % 3],
                domain=domain,
                domain_definition=domain_definition,
                parameters=self.parameter_values,
                func=self.func
            )
        return prompt

    def data_prompt(self):
        prompt = data_message.format(
            fewshot=self.fewshot,
            simple_agents=self.simple_agent_prompt,
            max_msg=MAX_MSG,
            round=self.round,
            func=self.func,
            parameter_values=self.parameter_values,
            agents=self.orchestrator_agent_prompt,
            task_desc=task_desc[self.task]
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
