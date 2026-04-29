import json
import ast
import re

with open('', 'r') as f_in, \
    open('', 'w') as f_out:
    data = [json.loads(line) for line in f_in.readlines()]

    for d in data:
        system = d['system']
        tools = d['tools']
        conversations = d['conversations']
        new_conversations = []
        for c in conversations:
            if c['from'] == 'user':
                new_conversations.append({'from': 'user', 'value': c['value']})
            elif c['from'] == 'assistant' and "<tool_calls>" not in c['value']:
                new_conversations.append({'from': 'assistant', 'value': c['value']})
            elif c['from'] == 'assistant' and "<tool_calls>" in c['value']:
                tool_calls = re.findall(r'<tool_calls>(.*?)</tool_calls>', c['value'], re.DOTALL)
                content = re.findall(r'^.*?<tool_calls>(.*?)</tool_calls>', c['value'], re.DOTALL)
                tool_calls = ast.literal_eval(tool_calls[0])
                tool_calls_str = ''
                for tool_call in tool_calls:
                    function = tool_call['function']
                    name = function['name']
                    arguments = json.loads(function['arguments'])
                    tool_calls_str = tool_calls_str + "<tool_call>" + json.dumps({'name': name, 'arguments': arguments}) + "</tool_call>"
                new_conversations.append({'from': 'assistant', 'value': content[0] + tool_calls_str})
            elif c['from'] == 'tool':
                new_conversations.append({'from': 'tool', 'value': c['value']})

        f_out.write(json.dumps({
            'system': system,
            'tools': tools,
            'conversations': new_conversations
        }) + '\n')
