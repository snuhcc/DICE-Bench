
inference_prompt = """# TOOL DOCUMENT #:
{{tool_document}}

# GOAL #: 
Using the tools provided in # TOOL DOCUMENT #, generate the required functions and their parameter values to fully address the users' request in # DIALOGUE #, especially the last turn of what the user said. 
Only produce output in the JSON format specified below. **Do not include any additional text, explanations, or content outside the JSON format.**

# OUTPUT FORMAT #: 
```
{
  "functions": [
    {
      "function_name": "name_of_the_function",
      "function_reasoning": "reason_for_selecting_the_function",
      "parameters": [
        {
          "parameter_name": "name_of_the_parameter",
          "parameter_value": "value_of_the_parameter",
          "parameter_type": "type_of_the_parameter (e.g., string, integer, boolean, etc.)",
          "parameter_reasoning": "reason_for_selecting_the_parameter"
        }
      ]
    }
  ]
}
```

# REQUIREMENTS #: 
1. Each function must directly address the users' request in # DIALOGUE # and align with tools in the # TOOL DOCUMENT #.
2. Provide the actual values for the parameters, and functions based on the context of the dialogue whenever possible.
3. Define each parameter with its name, value, type, and a brief explanation of its role directly under "parameter_reasoning".
4. **Only generate output in the JSON format specified in # OUTPUT FORMAT #. Do not include any explanations, additional comments, or additional text outside the JSON.**
5. If there are no functions or parameters required based on the dialogue, return an empty JSON object: `{ "functions": [] }`.

# DIALOGUE #: 
{{dialogue}}

Now generate your result using the specified format excluding any additional text, explanations, or content outside the JSON format:
# RESULT #:
"""

tokenizer_template = """
{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- set date_string = "26 Jul 2024" %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{# System message 처리 #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{{- system_message }}
{{- "<|eot_id|>" }}

{# Messages 처리 (name 포함) #}
{%- for message in messages %}
    {%- if message.name is defined %}
        {{- "<|start_header_id|>" + message.role + "<|end_header_id|>\n\n" + message.name + " (" + message.role + "): " + message.content|trim + "<|eot_id|>" }}
    {%- else %}
        {{- "<|start_header_id|>" + message.role + "<|end_header_id|>\n\n" + message.content|trim + "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}

{# Assistant 응답 생성 지점 #}
{%- if add_generation_prompt %}
    {{- "<|start_header_id|>assistant<|end_header_id|>\n\n" }}
{%- endif %}
"""