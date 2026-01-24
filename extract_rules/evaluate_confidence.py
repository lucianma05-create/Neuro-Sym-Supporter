import json
import time
import requests
import os
import re

# =========================
# 状态定义
# =========================
# 包含 Seeker 自身状态和 Supporter 对问题的探索程度
ALL_STATES = {
    "no_insight": "The user lacks insight into the emotional and situational factors contributing to their current state.",
    "no_effective_action": "The user has not identified or taken any effective actions to address their issues.",
    "need_information": "The user requires additional information or clarification regarding their situation or available resources.",
    "need_trauma_processing": "The user may benefit from processing past traumatic experiences.",
    "unclear_emotion": "The user's emotional state is ambiguous or not clearly expressed.",
    "panic": "The user is experiencing panic or acute anxiety.",
    "explored_issue": "The supporter has already explored the core issue with the user."
}

# =========================
# 工具函数
# =========================
def build_dialogue_context(dialog, cur_idx):
    lines = []
    for i in range(cur_idx + 1):
        speaker = dialog[i]["speaker"]
        content = dialog[i]["content"].strip()
        lines.append(f"{speaker.upper()}: {content}")
    return "\n".join(lines)


def call_llm(prompt, api_key, model):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert in emotional support dialogue analysis."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    resp = requests.post("", headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_score(text):
    # 使用正则表达式提取数字，防止 LLM 输出多余文字
    match = re.search(r"([01](\.\d)?)", text.strip())
    if match:
        score = float(match.group(1))
        return round(min(max(score, 0.0), 1.0), 1)
    raise ValueError(f"Invalid score output: {text}")


def score_with_retry(prompt, label, api_key, model, max_retry):
    for attempt in range(1, max_retry + 1):
        try:
            return parse_score(call_llm(prompt, api_key, model))
        except Exception as e:
            print(f"  ⚠️  {label} failed, retrying ({attempt}/{max_retry})... Error: {e}")
            if attempt == max_retry:
                return 0.0 # 最终失败则赋默认值 0.0
            time.sleep(1)


def build_prompt(dialogue, state_name, state_def):
    return f"""
You are given a multi-turn emotional support dialogue.

Dialogue history:
{dialogue}

State to evaluate:
{state_name}: {state_def}

Task:
Evaluate how strongly the CURRENT turn (the last line in history) satisfies the state above.
Note: If evaluating 'explored_issue', consider if the Supporter's previous efforts or the Seeker's current disclosure indicate the issue is now well-understood.

Scoring rule:
- 0.0 = does not satisfy at all
- 1.0 = fully satisfies
- Use one decimal place only

Output:
Return ONLY a single number (e.g., 0.0, 0.5, 1.0).
"""


# =========================
# 主函数
# =========================
def score_dialogues(input_path, output_path, api_key, model, max_retry=3):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 统计 Seeker 的总轮数用于进度显示
    total_seeker_turns = sum(1 for item in data for turn in item.get("dialog", []) if turn.get("speaker") == "seeker")
    processed_seeker = 0

    for d_idx, item in enumerate(data):
        dialog = item.get("dialog", [])
        dialog_len = len(dialog)

        for t_idx, turn in enumerate(dialog):
            speaker = turn.get("speaker")
            
            # 只在 Seeker 轮次进行所有状态的评分并写入
            if speaker == "seeker":
                processed_seeker += 1
                context = build_dialogue_context(dialog, t_idx)
                
                print(f"\n[Dialog {d_idx+1} | Seeker Turn {processed_seeker}/{total_seeker_turns}]")
                
                symbolic_state = {}
                for state, definition in ALL_STATES.items():
                    print(f"  → Scoring symbolic_state: {state} ... ", end="")
                    prompt = build_prompt(context, state, definition)
                    score = score_with_retry(prompt, state, api_key, model, max_retry)
                    symbolic_state[state] = score
                    print(score)

                # 写入到 Seeker 的 annotation
                turn.setdefault("annotation", {})
                turn["annotation"]["symbolic_state"] = symbolic_state

                progress = processed_seeker / total_seeker_turns * 100
                print(f"Seeker progress: {progress:.1f}%")
                print("-" * 50)

    # 确保保存路径存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 全部完成，结果已保存至 {output_path}")


# =========================
# 执行入口
# =========================
if __name__ == "__main__":
    score_dialogues(
        input_path="../data/ESConv/ESConv_merged.json",
        output_path="../data/ESConv/ESConv_with_symbolic_state.json",
        api_key="sk-xxxx", 
        model="gpt-4o-mini",
        max_retry=3
    )