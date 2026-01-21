import json
import os
from typing import Any, List, Union


def is_strict_int(x: Any) -> bool:
    # bool 是 int 的子类，必须排除
    return isinstance(x, int) and not isinstance(x, bool)


def is_pair(node: Any) -> bool:
    """
    严格匹配规则：
    - node 是 list
    - 长度恰好为 2
    - 两个元素都是 int（非 bool）
    - a >= 1 且 b >= 1
    """
    if not isinstance(node, list):
        return False
    if len(node) != 2:
        return False
    a, b = node
    if not (is_strict_int(a) and is_strict_int(b)):
        return False
    return a >= 1 and b >= 1


def _path_join(parent: str, key: Union[str, int]) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    if parent == "$":
        return f"$.{key}"
    return f"{parent}.{key}"


def _transform(node: Any, path: str, replaced_paths: List[str]) -> Any:
    """
    递归遍历 dict / list，并在满足 is_pair 时进行替换
    """
    if is_pair(node):
        a, b = node
        replaced_paths.append(path)
        return float(a) / float(b)

    if isinstance(node, dict):
        return {
            k: _transform(v, _path_join(path, k), replaced_paths)
            for k, v in node.items()
        }

    if isinstance(node, list):
        return [
            _transform(v, _path_join(path, i), replaced_paths)
            for i, v in enumerate(node)
        ]

    return node


def replace_pairs_in_json_file(f_in: str) -> str:
    """
    参数
    ----
    f_in : str
        输入 JSON 文件路径

    返回
    ----
    f_out : str
        输出 JSON 文件路径（与原文件同目录，文件名为 <原名>.out.json）
    """
    if not os.path.isfile(f_in):
        raise FileNotFoundError(f"Input file not found: {f_in}")

    try:
        with open(f_in, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse failed: {e}") from e

    replaced_paths: List[str] = []
    new_data = _transform(data, "$", replaced_paths)

    base, ext = os.path.splitext(f_in)
    f_out = f"{base}.out{ext or '.json'}"

    with open(f_out, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 控制台输出替换次数（如需路径，可在此打印 replaced_paths）
    print(len(replaced_paths))

    return f_out
