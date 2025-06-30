from src.prompt.domain_prompt import domain_prompt_dict


basic_system_message = """
You are a cooperative AI assistant participating in a multi-agent system. You collaborate with other user agents and an orchestrator to generate a purposeful, contextually relevant conversation.

- Your primary goals:
1) **Conversational Quality**:
   - Keep the conversation logically coherent and natural across all turns.
   - Incorporate parameter values smoothly into the context.
   - Avoid any GPT error messages or refusals.
   - Maintain a consistent style/tone matching the dialogue's domain and each agent's persona.

2) **Functional Integration**:
   - Call the AI Assistant every round with a clear, logically valid reason.
   - Use the previous round's return value correctly in the next round.
   - Ensure function name and parameters are inferable from context.
   - Align the AI's responses with the user's intent.

3) **Real-World Applicability**:
   - Function names and parameters should map to plausible real-world APIs.
   - The conversation content and function calls should feel authentic and realistically motivated.

4) **Must strictly adhere to the domain dialogue domain definition.**

Follow these points to keep the dialogue purposeful, natural, and consistent throughout all rounds.
"""


agent_system_message = """
{persona}
As a user agent in the "{domain}" domain:
- Future dialogues must be designed to strictly adhere to the domain definitions provided below.
{domain_definition}
- Stay consistent with your persona (tone, style, reasoning).
- Use only one short sentence per turn.
- Avoid directly mentioning function names in your response.
- **Do not attempt to call or request any AI function. Engage in discussion and gather enough context first.**
- **Do not generate [NEXT: ...] in your response.**
"""


orchestrator_system_message = """
You are the orchestrator managing a multi-agent conversation.

1) In each response, you must output exactly one of the following (and nothing else):
   {agents}
   "[NEXT: END]"

2) Use the format: [NEXT: agent_a]
   - No extra text or explanation beyond this bracketed command.

3) Select which agent speaks next based on:
   - The conversation's context,
   - The domain's requirements,
   - Varying the speaking order to avoid immediate repetition.

4) The conversation must have at least {max_msg} turns (excluding your own orchestrator messages) before you can choose "[NEXT: END]".

5) If an agent tries to call a function too early (before at least 8 turns), ignore it and continue letting them discuss. Only once there's sufficient context, at least {max_msg}+ turns have been reached, and you think conversation is repetitive, you may finalize with "[NEXT: END]".
"""


task_desc = {
    "single_round": """
The conversation to be generated is single-round.
- A "round" is a complete sequence of multi-party dialogue that ends with an AI function call.
- Ensure the conversation naturally leads to the given function and parameters.
- Keep the flow smooth, logical, and cohesive.
""",
    "multi_round": """
The conversation to be generated is multi-round.
- Each round ends with a function call.
- The next round must incorporate the previous round's virtual output or context.
- All rounds connect seamlessly for a coherent multi-round storyline.
- Make sure each function and its parameters are introduced organically and used appropriately.
""",
}


data_message = """
{fewshot}
<INSTRUCTION>
Carry out a natural, casual conversation with agents {simple_agents}, in random order decided by the orchestrator.

- {task_desc}
- At the conclusion, one agent should summarize key decisions and request the AI to use the required functions.
- "[NEXT: END]" is only possible after satisfying all requirements, including at least {max_msg} turns.
- All function parameters must appear naturally within the conversation flow.
- **Avoid any function calls before at least 8 agent turns** have happened in total, so the discussion has enough depth.
- The final agent's last utterance should address the AI to perform the function without including the information of both parameter values and function name.
  (e.g., "AI, please turn on the computer." instead of mentioning the function name or parameters).
- Stay logical, consistent, and goal-driven in each turn.
- Do not include both parameter values and function name in the same sentence.
"""


class PromptMaker:
    def __init__(
        self,
        agent_num,
        rounds_num,
        fewshot,
        function_dumps_per_dialogue,
        domain,
        task,
        personas,
        max_turns,
    ):
        self.agent_num = agent_num
        self.rounds_num = rounds_num
        self.fewshot = fewshot
        self.function_dumps_per_dialogue = function_dumps_per_dialogue
        self.domain = domain
        self.task = task
        self.personas = personas
        self.max_msg = max_turns

        self.agent_names = [f"agent_{chr(97 + i)}" for i in range(self.agent_num)]

        self.next_agent_list = "".join(
            [f"""'[NEXT: {self.agent_names[i]}]',""" for i in range(self.agent_num)]
        )

        self.simple_agent_prompt = " ".join(
            [self.agent_names[i][-1].capitalize() for i in range(self.agent_num)]
        )

    def agent_prompt(self, agent_type, agent_num):
        domain, domain_definition = self.get_domain()

        prompt = basic_system_message.format(domain_definition=domain_definition)

        if agent_type == "orch":
            prompt += orchestrator_system_message.format(
                agents=self.next_agent_list, max_msg=self.max_msg
            )
        else:
            persona_index = (ord(agent_type) - 97) % self.agent_num
            prompt += agent_system_message.format(
                persona=self.personas[persona_index],
                domain=domain,
                domain_definition=domain_definition,
            )

        return prompt

    def data_prompt(self):
        domain, domain_definition = self.get_domain()
        prompt = data_message.format(
            fewshot=self.fewshot,
            simple_agents=self.simple_agent_prompt,
            max_msg=self.max_msg,
            agents=self.next_agent_list,
            task_desc=task_desc[self.task],
            task=self.task,
            domain_definition=domain_definition,
        )
        return prompt

    def get_domain(self):
        return self.domain, domain_prompt_dict[self.domain]
