import random
from src.prompt.domain_prompt import domain_prompt_dict

## define agent prompt here
MAX_MSG = 10

# 어떤 페르소나에든 들어가는 default
basic_message = """
    You are a helpful AI assistant.\n
    - Collaborate in a discussion based on your role and the messages provided.\n
    - Respond according to your designated persona.\n
    - Provide only one short sentence per turn if you are an agent.\n
    - Try to make the conversation be approximately {max_msg} steps except for the utterances from "orchestrator".\n
    """

# 페르소나 정의
agent_system_message = """You are an Agent {agent_char}.
Try to progress the {domain} conversation where the {domain} is defined as the following:
{domain_definition}.
Make sure the conversation should be very naturally, and make sure the parameter values can be decided through the conversation that you will generate.
"""

# 3개 돌려쓰기.
agent_personas = [
    """a thoughtful and resourceful problem-solver who likes optimizing plans for the group's benefit. 
    You focus on finding the best options for costs, convenience, and logistics.""",
    """a detail-oriented and practical thinker who ensures that the plans are realistic and well-organized. 
    You focus on logistics like scheduling and timing, balancing fun with practicality.""",
    """a spontaneous and energetic planner who loves initiating plans and suggesting ideas.
    Your focus is on creating exciting plans and keeping the conversation dynamic and engaging."""
]

# TODO: graph based generation
orchestrator_system_message = (
    "You are the orchestrator, responsible for controlling the speaking order in this conversation. "
    "Generate '[NEXT: END]' after {max_msg} turns of utterances from agents"
    "Never add anything beyond calling agents or make the dialogue end."
    # "Within {max_msg} messages, ensure agent(s) make {round} function call(s) to the AI using the specified function(s). "
    # "Refer function list below:"
    # "{func_doc}"
    "Choose the next agent most relevant to the chat history and persona."
    "Avoid the same agent to speak consecutively."
    "Conclude the session with '[NEXT: END]'."
    "You must respond only with exactly one of the following: {agents} or '[NEXT: END]'. No additional text or explanations are allowed."
)


# function call output : GPT 생성 유도 (get_weather 같은거)
# topic: 좀더 구체화
data_message = """{fewshot}
    <INSTRUCTION>
    Carry out a natural and casual conversation similar to everyday life scenarios.  
    Ensure the conversation flows smoothly, with agents {simple_agents} speaking in random order and never repeating consecutively. Use orchestrator to determine speaker.
    The conversation must explicitly mention all parameter values provided in {func_doc} in a natural and logical manner during the discussion. These values should contribute meaningfully to the flow of the conversation and should not feel forced or out of place.
    The orchestrator will conclude the conversation with '[NEXT: END]' after all conditions are met.
    The last utterance from the last agent should say something that calls AI, for example, "Hey, AI, please handle our issue" or "AI, go ahead" etc.
    All conversation need to be in Korean.
"""
# Within {max_msg} messages, naturally incorporate all the details needed to fulfill {round} number function call(s) during the dialogue. Earlier function calls' outputs need to be reflect to later dialogue. The latest function call need to be called in last message.
    

class PromptMaker:
    def __init__(self, agent_num, round, fewshot, funclist, domain,iter):
        self.agent_num = agent_num
        self.round = round
        self.fewshot = fewshot
        self.func_doc = funclist.func_doc

        self.domain = domain
        self.now_domains = [self.get_domain() for i in range(iter)]
        self.domain_ctr = 0

        agent_names = [f'agent_{chr(97+i)}' for i in range(agent_num)]
        self.personas = [agent_personas[i%3] for i in range(agent_num)]
        self.orchestrator_agent_prompt = "".join([f"""'[NEXT: {agent_names[i]}]',""" for i in range(agent_num) ])
        self.simple_agent_prompt = " ".join([agent_names[i][-1].capitalize() for i in range(agent_num)])

    def agent_prompt(self, agent_type):
        prompt = basic_message.format(agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG)

        if agent_type == 'orch':
            prompt += orchestrator_system_message.format(agents=self.orchestrator_agent_prompt, max_msg=MAX_MSG, round=self.round)
        else:
            domain, domain_prompt = self.now_domains[self.domain_ctr]
            self.domain_ctr +=1
            prompt += agent_system_message.format(agent_char=agent_type, persona=agent_personas[(ord(agent_type)-97) % 3], domain=domain, domain_definition=domain_prompt)
        return prompt
    
    def data_prompt(self):
        prompt = data_message.format(fewshot=self.fewshot, simple_agents=self.simple_agent_prompt, max_msg=MAX_MSG, round=self.round, func_doc=self.func_doc, )
        return prompt

    def get_domain(self):
        if self.domain == 'Random':
            r_domain = random.choice([
                "Persuassion", "Negotiation", "Inquiry", "Deliberation", "Information-Seeking", "Eristic"
            ])
            return r_domain, domain_prompt_dict[r_domain]
        else:
            return self.domain, domain_prompt_dict[self.domain]