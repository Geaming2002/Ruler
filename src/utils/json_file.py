import json


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def load_jsonl(path):
    data = []
    with open(path, "r") as file:
        for line in file:
            json_data = json.loads(line)
            data.append(json_data)
    return data


def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as file:
        for item in data:
            json_string = json.dumps(item, ensure_ascii=False)
            file.write(json_string + "\n")
