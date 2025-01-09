import json
from pathlib import Path
import os

with open('./tool_graph.json', 'r') as f:
    data = json.load(f)
print(len(data['links']))