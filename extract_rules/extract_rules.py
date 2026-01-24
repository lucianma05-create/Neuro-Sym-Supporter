import json
from collections import defaultdict

# ========== 全局变量 ==========
THRESHOLD = 0.4      # symbolic_state 阈值
MIN_COUNT = 35       # 出现次数最小阈值
INPUT_FILE = "../data/ESConv/ESConv_with_symbolic_state.json"
OUTPUT_FILE = "../data/ruleset.json"

def extract_rules(input_file=INPUT_FILE, output_file=OUTPUT_FILE,
                  threshold=THRESHOLD, min_count=MIN_COUNT):
    """
    从对话数据中提取 symbolic_state + strategy 组合的规则
    并统计每条规则出现次数，只保留 count >= min_count 的规则
    输出 JSON 格式：
    {
        "策略1": [{"symbolic_state": [...], "count": N}, ...],
        "策略2": [...]
    }
    """
    # 读取原始对话数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 用于统计规则出现次数，key = (strategy, frozenset(symbolic_state))
    rule_counter = defaultdict(int)

    # 遍历每段对话
    for dialog_data in data:
        dialog = dialog_data.get("dialog", [])

        # 遍历对话中的每一句话
        for i, turn in enumerate(dialog):
            if turn.get("speaker") != "seeker":
                continue

            annotation = turn.get("annotation", {})
            raw_text_state = annotation.get("text_state", {})

            # symbolic_state = 大于阈值的状态键集合
            symbolic_state = sorted([k for k, v in raw_text_state.items() if v >= threshold])
            if not symbolic_state:
                continue  # 没有达到阈值的状态直接跳过

            # 找到该 seeker 发言后最近的 supporter 的策略
            strategy_list = []
            for j in range(i + 1, len(dialog)):
                if dialog[j].get("speaker") == "supporter":
                    strategy_list = dialog[j].get("annotation", {}).get("strategy", [])
                    break
            if not strategy_list:
                continue  # 没有找到策略，跳过

            # 每个策略都生成一条规则
            for strat in strategy_list:
                key = (strat, frozenset(symbolic_state))
                rule_counter[key] += 1

    # 聚合规则，按策略名分组
    rules_by_strategy = defaultdict(list)
    for (strategy, sym_state_set), count in rule_counter.items():
        if count < min_count:
            continue  # 小于最小计数的规则不保存
        rules_by_strategy[strategy].append({
            "symbolic_state": sorted(list(sym_state_set)),
            "count": count
        })

    # 保存到 JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(rules_by_strategy, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    total_rules = sum(len(v) for v in rules_by_strategy.values())
    print(f"规则提取完成，共 {total_rules} 条规则，已保存到 {output_file}")

# ========== 可作为独立脚本运行 ==========
if __name__ == "__main__":
    extract_rules()
