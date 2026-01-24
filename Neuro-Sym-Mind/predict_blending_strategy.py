import json
import torch
import os
from train_blend import load_models, blend_predict, get_neural_strategy_names
from splite_data import prepare_and_load_datasets

# ========== 1. 配置路径 ==========
RAW_DATA_PATH = "../data/ESConv/ESConv_merged.json" 
SYM_MODEL_PATH = "./models/strategy_model.pt"
NN_MODEL_PATH = "./models/best_neural_strategy_predictor.pth"
A_NET_PATH = "./models/best_a_net.pth"  
SAVE_PATH = "./data/test_set_predicted.json" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ========== 2. 数据切分 (只取测试集) ==========
    # 这样能保证这里的 test_set 和你训练 blend 时看到的完全一致
    _, _, test_set = prepare_and_load_datasets(RAW_DATA_PATH)
    
    with open("./data/ruleset.json", "r", encoding="utf-8") as f:
        ruleset = json.load(f)

    # ========== 3. 加载模型 ==========
    blend_names = get_neural_strategy_names()
    symbolic_model, neural_model = load_models(ruleset, SYM_MODEL_PATH, NN_MODEL_PATH)
    
    # 关键：加载训练好的 a_net 参数
    if os.path.exists(A_NET_PATH):
        neural_model.a_net.load_state_dict(torch.load(A_NET_PATH, map_location=device))
        print(f"成功加载训练好的 a_net 权重: {A_NET_PATH}")
    else:
        print("警告: 未找到训练好的 a_net 权重，将使用初始权重。")

    neural_model.to(device)
    neural_model.eval()

    # ========== 4. 仅为测试集进行预测 ==========
    print(f"正在为测试集 ({len(test_set)} 条对话) 生成预测策略...")
    
    for item in test_set:
        dialog = item["dialog"]
        for i in range(len(dialog)):
            turn = dialog[i]
            # 目标是预测 supporter 的策略
            if turn["speaker"] == "supporter":
                # 寻找该 turn 之前的最近一条 seeker 发言
                seeker_turn = None
                for j in range(i - 1, -1, -1):
                    if dialog[j]["speaker"] == "seeker":
                        seeker_turn = dialog[j]
                        break
                
                if seeker_turn:
                    # 使用 blend_predict 进行推理
                    # 注意：函数签名需与你 train_blend.py 中的保持一致
                    # a 值现在由 neural_model.a_net 自动计算
                    blend_strategy, _, _, _, _, a_val = blend_predict(
                        seeker_turn, ruleset, symbolic_model, neural_model, device, blend_names
                    )
                    
                    # 写入结果到 annotation 字段
                    if "annotation" not in turn:
                        turn["annotation"] = {}
                    
                    # 保存预测的策略名和当时的 a 权重（方便分析）
                    turn["annotation"]["pre_strategy"] = [blend_strategy]
                    turn["annotation"]["blend_weight_a"] = float(a_val)

    # ========== 5. 保存为新文件 ==========
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！测试集预测结果已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()