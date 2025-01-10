import random
from src.prompt.domain_prompt import domain_prompt_dict

## define agent prompt here
MAX_MSG = 20

# 어떤 페르소나에든 들어가는 default
basic_message = """
    You are a helpful AI assistant.\n
    - Collaborate in a discussion based on your role and the messages provided.\n
    - Respond according to your designated persona.\n
    - Provide only one short sentence per turn if you are an agent.\n
    - If you are the orchestrator, you must respond only with exactly one of the following: {agents} or '[NEXT: END]'. No additional text or explanations are allowed.\n
    - End the entire conversation after {max_msg} total messages by prompting one agent to call AI using the functions in the functions document, ensuring all necessary information for the function call is included within the dialogue.\n
    """

# 페르소나 정의
agent_system_message = "You are User {agent_char}. {persona}"

# 3개 돌려쓰기.
agent_personas = [
    """a thoughtful and resourceful problem-solver who likes optimizing plans for the group's benefit. 
    You focus on finding the best options for costs, convenience, and logistics.""",
    """a detail-oriented and practical thinker who ensures that the plans are realistic and well-organized. 
    You focus on logistics like scheduling and timing, balancing fun with practicality.""",
    """a spontaneous and energetic planner who loves initiating plans and suggesting ideas.
    Your focus is on creating exciting plans and keeping the conversation dynamic and engaging."""
]

orchestrator_system_message = (
    "You are the orchestrator, responsible for controlling the speaking order in this conversation. "
    "Your valid responses are strictly limited to selecting one agent from {agents} or responding with '[NEXT: END]'. "
    "Never add anything beyond these responses. You cannot call function calls directly."
    "Within {max_msg} messages, ensure agent(s) make {round} function call(s) to the AI using the specified function(s). "
    "Maintain a random speaking order without allowing the same agent to speak consecutively, except when the final AI call is made. "
    "Conclude the session with '[NEXT: END]'."
)


# function call output : GPT 생성 유도 (get_weather 같은거)
# topic: 좀더 구체화
data_message = """{fewshot}
    <INSTRUCTION>
    Carry out a natural and casual conversation similar to everyday life scenarios.  
    The goal is to write a natural conversation where multiple users interact in an {domain}. An {domain} is defined as:  
    '{domain_definition}'  
    Ensure the conversation flows smoothly, with agents {simple_agents} speaking in random order and never repeating consecutively. Use orchestrator to determine speaker.
    The conversation must explicitly mention all parameter values provided in {func_doc} in a natural and logical manner during the discussion. These values should contribute meaningfully to the flow of the conversation and should not feel forced or out of place.
    Within {max_msg} messages, naturally incorporate all the details needed to fulfill {round} number function call(s) during the dialogue. Earlier function calls' outputs need to be reflect to later dialogue. The latest function call need to be called in last message.
    A function call need to be called by only agent, not orchestrator. 
    At least one agent must make the AI function call using the specified functions_document: {func_doc}.  
    The orchestrator will conclude the conversation with '[NEXT: END]' after all conditions are met.
"""


class PromptMaker:
    def __init__(self, agent_num, round, fewshot, funclist, domain):
        self.agent_num = agent_num
        self.round = round
        self.fewshot = fewshot
        self.func_doc = funclist.func_doc

        self.domain = domain

        agent_names = [f'agent_{chr(97+i)}' for i in range(agent_num)]
        self.personas = [agent_personas[i%3] for i in range(agent_num)]
        self.orchestrator_agent_prompt = "".join([f"""'[NEXT: {agent_names[i]}]',""" for i in range(agent_num) ])
        self.simple_agent_prompt = " ".join([agent_names[i][-1].capitalize() for i in range(agent_num)])

    def agent_prompt(self, agent_type):
        prompt = basic_message.format(agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG)

        if agent_type == 'orch':
            prompt += orchestrator_system_message.format(agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG, round=self.round)
        else:
            prompt += agent_system_message.format(agent_char=agent_type, persona=agent_personas[(ord(agent_type)-97) % 3])
        return prompt
    
    def data_prompt(self):
        domain, domain_prompt = self.get_domain()
        prompt = data_message.format(fewshot=self.fewshot, simple_agents=self.simple_agent_prompt, max_msg=MAX_MSG, round=self.round, func_doc=self.func_doc, domain=domain, domain_definition=domain_prompt )
        return prompt

    def get_domain(self):
        if self.domain == 'Random':
            r_domain = random.choice([
                "Persuassion", "Negotiation", "Inquiry", "Deliberation", "Information-Seeking", "Eristic"
            ])
            return r_domain, domain_prompt_dict[r_domain]
        else:
            return self.domain, domain_prompt_dict[self.domain]