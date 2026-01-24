import json
import torch
import os
from train_nero_mind import NeuralStrategyPredictor
from splite_data import prepare_and_load_datasets

# ========== 1. 配置路径 ==========
RAW_DATA_PATH = "../data/ESConv/ESConv_merged.json" # 原始总数据
MODEL_PATH = "./models/best_neural_strategy_predictor.pth"
SAVE_PATH = "./data/test_set_nero_predicted.json" # 神经模型专属输出

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # ========== 2. 数据切分 (只取测试集) ==========
    # 确保测试集与 Blend 脚本、训练脚本完全一致
    _, _, test_set = prepare_and_load_datasets(RAW_DATA_PATH)
    
    # ========== 3. 加载神经模型 ==========
    predictor = NeuralStrategyPredictor()
    # 内部包含加载逻辑，确保路径正确
    predictor.load_model(MODEL_PATH)
    predictor.policy_net.to(device)
    predictor.state_extractor.to(device)
    predictor.eval()

    print(f"正在使用神经模型为测试集 ({len(test_set)} 条对话) 生成预测...")

    # ========== 4. 遍历测试集进行预测 ==========
    for item in test_set:
        dialog = item["dialog"]
        for i in range(len(dialog)):
            turn = dialog[i]
            
            if turn["speaker"] == "supporter":
                # 寻找前一条 seeker 发言
                seeker_turn = None
                for j in range(i - 1, -1, -1):
                    if dialog[j]["speaker"] == "seeker":
                        seeker_turn = dialog[j]
                        break
                
                if seeker_turn:
                    # 获取文本输入
                    seeker_text = seeker_turn.get("content", "")
                    
                    # 提取状态并预测
                    state = predictor.state_extractor(seeker_text).to(device)
                    with torch.no_grad():
                        # select_action 内部处理了维度和 Tensor 转换
                        action = predictor.select_action(state, training=False)
                        # 将索引转换为策略名称字符串
                        predicted_strategy = predictor.idx_to_strategy[action]
                    
                    # 写入结果
                    if "annotation" not in turn:
                        turn["annotation"] = {}
                    
                    # 使用 pre_nero_strategy 字段以示区分
                    turn["annotation"]["pre_nero_strategy"] = [predicted_strategy]

    # ========== 5. 保存结果 ==========
    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 神经模型预测完成！结果已保存至: {SAVE_PATH}")

if __name__ == "__main__":
    main()