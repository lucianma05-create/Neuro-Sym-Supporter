import json
import torch
import random
import os
from train_sym_mind import load_model as load_symbolic_model, matching_roles
from splite_data import prepare_and_load_datasets

# ========== 1. 配置路径 ==========
RAW_DATA_PATH = "../data/ESConv/ESConv_merged.json"
RULESET_PATH = "./data/ruleset.json"
SYM_MODEL_PATH = "./models/strategy_model.pt"
SAVE_PATH = "./data/test_set_sym_predicted.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ========== 2. 数据切分 (只取测试集) ==========
    _, _, test_set = prepare_and_load_datasets(RAW_DATA_PATH)
    
    with open(RULESET_PATH, "r", encoding="utf-8") as f:
        ruleset = json.load(f)

    # ========== 3. 加载正式符号模型 ==========
    # 假设你的 train_sym_mind.py 中有 load_model 函数
    if os.path.exists(SYM_MODEL_PATH):
        symbolic_model = load_symbolic_model(ruleset, SYM_MODEL_PATH)
        symbolic_model.to(device)
        symbolic_model.eval()
        print(f"成功加载符号模型权重: {SYM_MODEL_PATH}")
    else:
        print(f"错误: 未找到模型文件 {SYM_MODEL_PATH}")
        return

    print(f"正在使用符号模型 (Top-3采样) 为测试集 ({len(test_set)} 条对话) 生成预测...")

    # ========== 4. 遍历测试集进行预测 ==========
    for item in test_set:
        dialog = item.get("dialog", [])
        for i in range(len(dialog)):
            turn = dialog[i]
            
            if turn.get("speaker") == "supporter":
                # 寻找前一条 seeker 发言
                seeker_turn = None
                for j in range(i - 1, -1, -1):
                    if dialog[j].get("speaker") == "seeker":
                        seeker_turn = dialog[j]
                        break
                
                if seeker_turn:
                    # 调用你定义的匹配逻辑 (内部包含 Top-3 逻辑)
                    # 注意：如果 matching_roles 在 train_sym_mind 里没写 Top-3，则在这里处理
                    pre_strategy, _, _ = matching_roles(seeker_turn, ruleset, symbolic_model, top_k=3)
                    
                    # 写入结果
                    if "annotation" not in turn:
                        turn["annotation"] = {}
                    
                    # 使用 pre_sym_strategy 字段
                    turn["annotation"]["pre_sym_strategy"] = [pre_strategy]

    # ========== 5. 保存结果 ==========
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 符号模型预测完成！结果已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()