import json
import os

def write_jsonl(file_path: str, data: list, append: bool = False):
    mode = "a" if append else "w"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def read_jsonl(file_path: str) -> list:
    if not os.path.exists(file_path):
        return []

    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    return records