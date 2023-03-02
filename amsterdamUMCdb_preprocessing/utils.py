import json

with open('paths.json', 'r') as f:
    amsterdam_path = json.load(f)['data_dir']