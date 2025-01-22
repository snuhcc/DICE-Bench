validation_graph_prompt = """# TOOL DOCUMENT #:
{tool_document}

# GOAL #
Using the tools provided in # TOOL DOCUMENT #, generate the required functions and their parameter values to fully address the users' request in # DIALOGUE #, especially the last turn of what the user said per round.
For each round in # DIALOGUE #, create a separate function. Each function must correspond to one round, ensuring all dialogue rounds are addressed.
Only produce output in the JSON format specified below. **Do not include any additional text, explanations, or content outside the JSON format.**

# OUTPUT FORMAT #
[
    {{
        "function_name": "name_of_the_function",
        "function_reasoning": "reason_for_selecting_the_function",
        "parameters": [
            {{
                "parameter_name": "name_of_the_parameter",
                "parameter_value": "value_of_the_parameter",
                "parameter_type": "type_of_the_parameter (e.g., string, integer, boolean, etc.)",
                "parameter_reasoning": "reason_for_selecting_the_parameter"
            }}
        ]
    }},
]

# REQUIREMENTS #:
1. Each function must directly address the user's request in # DIALOGUE # and utilize the tools available in # TOOL DOCUMENT #.
2. Each function should correspond to the context and details of one specific dialogue round. Length of output list and dialogue round number must be equal.
3. Provide the actual values for the parameters and functions based on the context of the dialogue whenever possible.
4. Ensure the function names and parameters align with those listed in # TOOL DOCUMENT #.
5. If no functions are required based on the # TOOL DOCUMENT #, return a single object with "function_name" as NONE.
6. Only provide the output in the specified JSON format, strictly adhering to the requirements, do not include additional text (e.g., ```, ```json).
"""


validation_prompt_dict = {
    "Graph": validation_graph_prompt
}


