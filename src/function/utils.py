from function_schema import get_function_schema


def get_tool_schema(function):
    return {
        'type': 'function',
        'function': get_function_schema(function)
    }

def create_function_docs(functions):
    docs = []
    for fn in functions:
        # get_tool_schema를 통해 함수 스키마를 얻음
        schema = get_tool_schema(fn)
        # 함수 이름과 스키마를 문자열로 변환
        docs.append(f"=== {fn.__name__} ===\n{schema}")
    # 줄바꿈하여 모두 합침
    return "\n\n".join(docs)
