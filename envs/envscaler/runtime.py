from __future__ import annotations

import json
import re
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional


def init_env_class(env_class_code: str, env_class_name: str):
    module = types.ModuleType("dynamic_env")
    exec(env_class_code, module.__dict__)
    if not hasattr(module, env_class_name):
        raise ValueError(f"Class '{env_class_name}' not found in provided env_class_code.")
    return getattr(module, env_class_name)


def init_env_instance(env_class, init_config: Optional[dict] = None):
    init_config = deepcopy(init_config)
    try:
        if init_config and isinstance(init_config, dict):
            env_instance = env_class(init_config)
        else:
            env_instance = env_class({})
    except TypeError:
        env_instance = env_class()

    if init_config:
        for key, value in init_config.items():
            setattr(env_instance, key, value)

    return env_instance


def get_state_info(env_instance) -> dict:
    return deepcopy({
        k: v for k, v in vars(env_instance).items()
        if not (k.startswith("__") and k.endswith("__"))
    })


def run_check_function(func_code: str, init_state: dict, final_state: dict):
    exec_globals: Dict[str, Any] = {"__builtins__": __builtins__, "initial_state": deepcopy(init_state)}
    try:
        exec(func_code, exec_globals)
        if "check_func" not in exec_globals:
            return False, None, "Function 'check_func' not found."
        result = exec_globals["check_func"](final_state)
        if not isinstance(result, bool):
            return False, None, "Function did not return a boolean."
        return True, result, None
    except Exception as e:
        return False, None, str(e)


def calculate_reward(checklist_with_func: List[dict], init_state: dict, final_state: dict) -> float:
    if not checklist_with_func:
        return 0.0
    passed = 0
    for item in checklist_with_func:
        ok, result, _ = run_check_function(item["check_func"], init_state, final_state)
        if ok and result is True:
            passed += 1
    return round(passed / len(checklist_with_func), 4)


def parse_response(text: str):
    parse_success = True
    result: Dict[str, Any] = {"reasoning_content": None, "tool_calls": None, "content": None}
    text = (text or "").strip()

    think_match = re.search(
        r"(?:<|@)think(?:>|@)\s*(.*?)(?:<|@)/think(?:>|@)", text, re.DOTALL | re.IGNORECASE,
    )
    if think_match:
        result["reasoning_content"] = think_match.group(1).strip()
    elif re.search(r"(?:<|@)think(?:>|@)", text, re.IGNORECASE):
        parse_success = False
        result["reasoning_content"] = {"error": "Missing closing think tag"}

    tool_calls = list(re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL))
    tool_call_content = None
    if tool_calls:
        tool_call_match = tool_calls[0]
        tool_call_content = tool_call_match.group(1)
    else:
        if "<tool_call>" in text and "</tool_call>" not in text:
            parse_success = False
            result["tool_calls"] = [{"error": "Unclosed tool_call tag"}]

    if think_match and tool_call_content:
        result["content"] = text[think_match.end():tool_call_match.start()].strip()
    elif think_match and not tool_call_content:
        result["content"] = text[think_match.end():].strip()
    elif not think_match and tool_call_content:
        result["content"] = text[:tool_call_match.start()].strip()
    else:
        result["content"] = text.strip()

    if tool_call_content:
        try:
            tool_call_dict = json.loads(tool_call_content)
            missing = [f for f in ("name", "arguments") if f not in tool_call_dict]
            if missing:
                parse_success = False
                result["tool_calls"] = [{"error": f"Missing required field(s): {missing}", "raw": tool_call_dict}]
            else:
                result["tool_calls"] = [{"function": tool_call_dict}]
        except json.JSONDecodeError as e:
            parse_success = False
            result["tool_calls"] = [{"error": f"Failed to parse tool_call JSON: {e}", "raw": tool_call_content}]

    return parse_success, result


def parse_action(struct_response: dict):
    try:
        if struct_response.get("tool_calls"):
            action = deepcopy(struct_response["tool_calls"][0]["function"])
            if isinstance(action.get("arguments"), str):
                action["arguments"] = json.loads(action["arguments"])
        elif struct_response.get("content"):
            action = {"name": "chat_with_user", "arguments": {"content": struct_response.get("content")}}
        else:
            return False, {}
        return True, action
    except Exception:
        return False, {}


SYSTEM_PROMPT = (
    "You are a helpful assistant. When given a specific task, your goal is to complete it in an "
    "interactive environment by making step-by-step use of available tools. \n"
    "- Before completing the task, at each step, select a tool from the tool list and fill in all "
    "required parameters, making sure that the values are valid. Avoid making parallel tool calls "
    "in one step.\n"
    "- When you believe the task has been completed, respond only with 'Task Completed' to end the "
    "trajectory, without adding any other content or making any tool calls.\n"
    "- It is recommended to first call query tools to gather sufficient information, then use "
    "modification tools to complete the task. Adjust actions promptly based on the feedback from "
    "the environment, i.e., the tool results.\n"
)


def construct_env_introduction(environment_introduction: str, constraints_rules: List[str]) -> str:
    env_rule_str = ""
    for rule in constraints_rules or []:
        env_rule_str += "- " + rule + "\n"
    return (
        "# Environment Information\n\n"
        f"## Brief Introduction:  \n{environment_introduction}\n\n"
        f"## Environment Rules / Constraints:  \n{env_rule_str}"
    )


def merge_tools_into_system_prompt(system_prompt: Optional[str], tools: Optional[List[dict]]) -> str:
    if not tools:
        return system_prompt or ""
    out: List[str] = []
    if system_prompt:
        out.append(system_prompt)
        out.append("\n\n")
    out.append("# Tools\n\n")
    out.append("You may call one or more functions to assist with the user query.\n\n")
    out.append("You are provided with function signatures within <tools></tools> XML tags:\n")
    out.append("<tools>")
    for tool in tools:
        out.append("\n")
        out.append(json.dumps(tool, ensure_ascii=False))
    out.append("\n</tools>\n\n")
    out.append("For each function call, return a json object with function name and arguments "
               "within <tool_call>...</tool_call> tags:\n")
    out.append("<tool_call>\n")
    out.append('{"name": <function-name>, "arguments": <args-json-object>}\n')
    out.append("</tool_call>")
    return "".join(out)


def build_envscaler_system_prompt(environment_introduction: str,
                                  constraints_rules: List[str],
                                  tools: List[dict]) -> str:
    intro = construct_env_introduction(environment_introduction, constraints_rules)
    return merge_tools_into_system_prompt(SYSTEM_PROMPT + "\n\n" + intro, tools)


