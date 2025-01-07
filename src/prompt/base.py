## define agent prompt here.

basic_message = """
    You are a helpful AI assistant.\n
    - Collaborate in a discussion based on your role and the messages provided.\n
    - Respond according to your designated persona.\n
    - Provide only one short sentence per turn if you are an agent.\n
    - If you are the mediator, you must respond only with exactly one of the following: {agents} or '[NEXT: END]'. No additional text or explanations are allowed.\n
    - End the entire conversation after 12 total turns by prompting one agent to call AI using the functions in the functions document, ensuring all necessary information for the function call is included within the dialogue.\n
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
    "You are the Mediator, controlling who speaks next. "
    "Your only valid responses are {agents} or '[NEXT: END]'. "
    "Never speak more than that. "
    "After 12 turns, have one agent call the AI with the relevant function(s), then conclude with '[NEXT: END]'."
    "Ensure conversation order is random without letting the same agent speak consecutively except for the very last turn when the agent call AI."
)
data_message = """{fewshot}
    Instruction:
    Have a very natural and casual conversation, just as you would in everyday life.  
    Refer to the few-shot examples above for guidance on how to structure the dialogue.  
    Ensure the conversation progresses smoothly, with agents {simple_agents} speaking in random order and not repeating consecutively.  
    Within 12 turns, make sure all necessary details for a final AI function call are naturally established through the dialogue.  
    Finally, one agent must call AI using the functions_document: {func_doc}, and the conversation concludes with the mediator responding '[NEXT: END]'.
"""

class PromptMaker:
    def __init__(self, agent_num, fewshot, funclist):
        self.agent_num = agent_num
        self.fewshot = fewshot
        self.func_doc = funclist.func_doc

        agent_names = ['agent_a', 'agent_b', 'agent_c']
        self.mediator_agent_prompt = "".join([f"""'[NEXT: {agent_names[i]}]',""" for i in range(agent_num) ])
        self.simple_agent_prompt = " ".join([agent_names[i][-1].capitalize() for i in range(agent_num)])

    def agent_prompt(self, agent_type):
        prompt = basic_message.format(agents=self.mediator_agent_prompt)
        if agent_type == 'A':
            prompt += agent_a_system_message
        elif agent_type == 'B':
            prompt += agent_b_system_message
        elif agent_type == 'C':
            prompt += agent_c_system_message
        elif agent_type == 'M':
            prompt += mediator_system_message.format(agents=self.mediator_agent_prompt)
        return prompt
    
    def data_prompt(self):
        prompt = data_message.format(fewshot=self.fewshot, simple_agents=self.simple_agent_prompt, func_doc=self.func_doc)
        return prompt