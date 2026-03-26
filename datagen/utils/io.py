import json
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: Path | str) -> dict:
    path = Path(path)
    with open(path) as f:
        return yaml.safe_load(f)


def write_yaml(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def read_json(path: Path | str) -> dict | list:
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def write_json(data: Any, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def read_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(data: list[dict], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def ensure_dir(path: Path | str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
