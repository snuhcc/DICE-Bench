## define agent prompt here.

MAX_MSG = 12

basic_message = """
    You are a helpful AI assistant.\n
    - Collaborate in a discussion based on your role and the messages provided.\n
    - Respond according to your designated persona.\n
    - Provide only one short sentence per turn if you are an agent.\n
    - If you are the mediator, you must respond only with exactly one of the following: {agents} or '[NEXT: END]'. No additional text or explanations are allowed.\n
    - End the entire conversation after {max_msg} total messages by prompting one agent to call AI using the functions in the functions document, ensuring all necessary information for the function call is included within the dialogue.\n
    """

# 페르소나 정의
agent_a_system_message = (
    "You are User A, a spontaneous and energetic planner who loves initiating plans and suggesting ideas."
    "Your focus is on creating exciting plans and keeping the conversation dynamic and engaging."
)

agent_b_system_message = (
    "You are User B, a detail-oriented and practical thinker who ensures that the plans are realistic and well-organized. "
    "You focus on logistics like scheduling and timing, balancing fun with practicality."
)

agent_c_system_message = (
    "You are User C, a thoughtful and resourceful problem-solver who likes optimizing plans for the group's benefit. "
    "You focus on finding the best options for costs, convenience, and logistics."
)

mediator_system_message = (
    "You are the Mediator, responsible for controlling the speaking order in this conversation. "
    "Your valid responses are strictly limited to selecting one agent from {agents} or responding with '[NEXT: END]'. "
    "Never add anything beyond these responses. "
    "Within {max_msg} messages, ensure agent(s) make {turn} function call(s) to the AI using the specified function(s). "
    "Maintain a random speaking order without allowing the same agent to speak consecutively, except when the final AI call is made. "
    "Conclude the session with '[NEXT: END]'."
)
data_message = """{fewshot}
    Instruction:
    Carry out a natural and casual conversation similar to everyday life scenarios.  
    Use the few-shot examples above as a reference for structuring the dialogue.  
    Ensure the conversation flows smoothly, with agents {simple_agents} speaking in random order and never repeating consecutively.  
    Within {max_msg} messages, naturally incorporate all the details needed to fulfill {turn} function call(s) during the dialogue.  
    At least one agent must make the AI function call using the specified functions_document: {func_doc}.  
    The Mediator will conclude the conversation with '[NEXT: END]' after all conditions are met.
"""

class PromptMaker:
    def __init__(self, agent_num, turn, fewshot, funclist):
        self.agent_num = agent_num
        self.turn = turn
        self.fewshot = fewshot
        self.func_doc = funclist.func_doc

        agent_names = ['agent_a', 'agent_b', 'agent_c']
        self.mediator_agent_prompt = "".join([f"""'[NEXT: {agent_names[i]}]',""" for i in range(agent_num) ])
        self.simple_agent_prompt = " ".join([agent_names[i][-1].capitalize() for i in range(agent_num)])

    def agent_prompt(self, agent_type):
        prompt = basic_message.format(agents=self.mediator_agent_prompt, max_msg=MAX_MSG)
        if agent_type == 'A':
            prompt += agent_a_system_message
        elif agent_type == 'B':
            prompt += agent_b_system_message
        elif agent_type == 'C':
            prompt += agent_c_system_message
        elif agent_type == 'M':
            prompt += mediator_system_message.format(agents=self.mediator_agent_prompt, max_msg=MAX_MSG, turn=self.turn)
        return prompt
    
    def data_prompt(self):
        prompt = data_message.format(fewshot=self.fewshot, simple_agents=self.simple_agent_prompt, max_msg=MAX_MSG, turn=self.turn, func_doc=self.func_doc)
        return prompt