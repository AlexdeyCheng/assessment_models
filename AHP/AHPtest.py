import json,math,numpy as np
from pyahp import parse
import clearjson 


out_path = clearjson.replace_pairs_in_json_file("examples/television.json")

with open(out_path, "r", encoding="utf-8") as f:
    model = json.load(f)   

ahp_model = parse(model)
priorities = ahp_model.get_priorities()
print(priorities)