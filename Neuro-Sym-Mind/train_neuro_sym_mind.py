import torch
import numpy as np
import json
import sys
import random
import logging
import torch.nn as nn
from train_sym_mind import load_model as load_symbolic_model, matching_roles
from train_neuro_mind import NeuralStrategyPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt

# 引入数据集划分模块
from splite_data import prepare_and_load_datasets

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 策略空间统一 ===========
def get_neural_strategy_names():
    return [
        "Question", "Affirmation and Reassurance", "Information",
        "Self-disclosure", "Providing Suggestions", "Restatement or Paraphrasing",
        "Others", "Reflection of feelings"
    ]

def align_probs(probs, src_names, target_names):
    out = np.zeros(len(target_names), dtype=np.float32)
    for i, name in enumerate(src_names):
        if name in target_names:
            idx = target_names.index(name)
            out[idx] = probs[i]
    return out

# ========== 加载模型 ===========
import torch.nn as nn

def load_models(ruleset, sym_model_path, nn_model_path):
    # 1. 加载符号模型
    symbolic_model = load_symbolic_model(ruleset, sym_model_path)
    
    # 2. 初始化神经端
    neural_model = NeuralStrategyPredictor()
    device = neural_model.device
    
    # 3. 加载权重
    logger.info(f"Loading neural checkpoint: {nn_model_path}")
    checkpoint = torch.load(nn_model_path, map_location=device)
    
    # 获取真正的 state_dict 数据
    if isinstance(checkpoint, dict) and 'policy_net_state_dict' in checkpoint:
        actual_weights = checkpoint['policy_net_state_dict']
    else:
        actual_weights = checkpoint

    # --- 核心修改：对整个 neural_model 进行非严格加载 ---
    # 这样 PyTorch 会自动根据 key (如 "policy_net.xxx") 分发到子模块
    # 如果 actual_weights 里的 key 不带 "policy_net." 前缀，我们需要手动给它加上
    
    first_key = next(iter(actual_weights))
    if not first_key.startswith('policy_net.') and not first_key.startswith('a_net.'):
        # 说明是纯 DQN 权重，给它包一层前缀以便父模型识别
        logger.info("Adding 'policy_net.' prefix to state_dict keys...")
        actual_weights = {f"policy_net.{k}": v for k, v in actual_weights.items()}

    # 重点：这里用 neural_model 而不是 neural_model.policy_net
    neural_model.load_state_dict(actual_weights, strict=False)
    logger.info("Successfully loaded weights into neural_model (strict=False).")

    # 4. 初始化 a_net 的权重和偏置
    # 注意：如果你的 a_net 层级不同，这里可能会报错，请根据实际情况调整索引
    with torch.no_grad():
        try:
            # 尝试访问 Sequential 的最后一层 Linear
            target_layer = None
            for layer in reversed(neural_model.a_net):
                if isinstance(layer, nn.Linear):
                    target_layer = layer
                    break
            
            if target_layer:
                nn.init.constant_(target_layer.bias, 1.0) # 初始 a ≈ 0.73
                nn.init.xavier_uniform_(target_layer.weight)
                logger.info(f"Initialized a_net bias to 1.0 (Initial a ≈ 0.73)")
        except Exception as e:
            logger.warning(f"Manual a_net initialization failed (not critical): {e}")

    neural_model.eval()
    return symbolic_model, neural_model

# ========== blend推理 ===========
def blend_predict(seeker_dialog, ruleset, symbolic_model, neural_model, device, blend_names=None):
    # 1. 符号端
    _, sym_scores, sym_names = matching_roles(seeker_dialog, ruleset, symbolic_model)
    sym_probs = sym_scores.detach().cpu().numpy()
    
    # 2. 神经端
    text = seeker_dialog["annotation"]["symbolic_state"].get("utter", seeker_dialog.get("content", ""))
    with torch.no_grad():
        state = neural_model.state_extractor(text).to(device)
        q_values = neural_model.policy_net(state)
        nn_probs = torch.softmax(q_values, dim=1).cpu().numpy().flatten()
        # a 作为一个标量
        a = neural_model.a_net(state).item()
    
    if blend_names is None:
        blend_names = get_neural_strategy_names()
    
    # 对齐并平滑 (加入 1e-6 防止 0 概率导致 log 溢出)
    sym_p = align_probs(sym_probs, sym_names, blend_names) + 1e-6
    nn_p = align_probs(nn_probs, get_neural_strategy_names(), blend_names) + 1e-6
    
    sym_p /= sym_p.sum() 
    nn_p /= nn_p.sum()
    
    blend_probs = a * sym_p + (1 - a) * nn_p
    blend_idx = np.argmax(blend_probs)
    
    return blend_names[blend_idx], blend_probs, sym_p, nn_p, blend_names, a

# ========== 训练 a 网络 ===========
def optimize_a_net(train_data, valid_data, ruleset, symbolic_model, neural_model, device, blend_names, num_epochs=1):
    # 只优化 a_net 的参数
    optimizer = torch.optim.Adam(neural_model.a_net.parameters(), lr=0.0001)
    best_val_acc = 0.0
    a_history = []

    for epoch in range(num_epochs):
        neural_model.a_net.train()
        total_loss = 0
        total_a = 0
        count = 0
        
        random.shuffle(train_data)
        pbar = tqdm(train_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for item in pbar:
            dialog = item["dialog"]
            for i in range(len(dialog) - 1):
                if dialog[i]["speaker"] == "seeker" and dialog[i+1]["speaker"] == "supporter":
                    # 获取真实标签
                    real_strategy_list = dialog[i+1]["annotation"]["strategy"]
                    if not real_strategy_list: continue
                    
                    # 准备输入
                    text = dialog[i]["annotation"]["symbolic_state"].get("utter", dialog[i].get("content", ""))
                    state = neural_model.state_extractor(text).to(device)
                    
                    # 计算当前 a 
                    a = neural_model.a_net(state).squeeze()
                    
                    # 预获取对齐后的两端分布（转换为 tensor 参与计算）
                    # 为了效率，在此处再次调用 matching_roles 的逻辑
                    _, _, sym_p_np, nn_p_np, _, _ = blend_predict(dialog[i], ruleset, symbolic_model, neural_model, device, blend_names)
                    
                    sym_p = torch.tensor(sym_p_np, device=device)
                    nn_p = torch.tensor(nn_p_np, device=device)
                    
                    # 混合分布
                    blend_p = a * sym_p + (1 - a) * nn_p
                    
                    # 构建目标 (label smoothing)
                    target = torch.zeros_like(blend_p)
                    for s in real_strategy_list:
                        if s in blend_names:
                            target[blend_names.index(s)] = 1.0 / len(real_strategy_list)
                    
                    # NLL Loss
                    loss = -torch.sum(target * torch.log(blend_p + 1e-8))
                    
                    optimizer.zero_grad()
                    loss.backward()
                    # 梯度裁剪防止抖动
                    torch.nn.utils.clip_grad_norm_(neural_model.a_net.parameters(), 1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    total_a += a.item()
                    count += 1
                    
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Mean_a": f"{(total_a/count if count > 0 else 0):.4f}"
            })

        # 验证
        val_acc = evaluate_blend(valid_data, ruleset, symbolic_model, neural_model, device, blend_names)
        print(f"[*] Epoch {epoch+1} Finished | Avg Loss: {total_loss/count:.4f} | Val Acc: {val_acc:.4f} | Final Mean a: {total_a/count:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(neural_model.a_net.state_dict(), "../models/best_a_net.pth")
            print(">> Best a_net Model Saved.")

# ========== 评估函数 ===========
def evaluate_blend(data, ruleset, symbolic_model, neural_model, device, blend_names):
    neural_model.a_net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for item in data:
            dialog = item["dialog"]
            for i in range(len(dialog) - 1):
                if dialog[i]["speaker"] == "seeker" and dialog[i+1]["speaker"] == "supporter":
                    strategy, _, _, _, _, _ = blend_predict(dialog[i], ruleset, symbolic_model, neural_model, device, blend_names)
                    if strategy in dialog[i+1]["annotation"]["strategy"]:
                        correct += 1
                    total += 1
    return correct / total if total > 0 else 0.0

# ========== 主程序 ===========
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sym_model', type=str, default='../models/sym_mind.pt')
    parser.add_argument('--nn_model', type=str, default='../models/neuro_mind.pth')
    parser.add_argument('--raw_data', type=str, default='../data/ESConv/ESConv_with_symbolic_state.json')
    parser.add_argument('--ruleset', type=str, default='../data/ruleset/ruleset.json')
    parser.add_argument('--train_a', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, valid_set, test_set = prepare_and_load_datasets(args.raw_data)
    
    with open(args.ruleset, 'r', encoding='utf-8') as f:
        ruleset = json.load(f)
    blend_names = get_neural_strategy_names()

    symbolic_model, neural_model = load_models(ruleset, args.sym_model, args.nn_model)
    neural_model.to(device)

    if args.train_a:
        optimize_a_net(train_set, valid_set, ruleset, symbolic_model, neural_model, device, blend_names)

    # 加载最优并测试
    if torch.os.path.exists("../models/best_a_net.pth"):
        neural_model.a_net.load_state_dict(torch.load("../models/best_a_net.pth"))
    
    test_acc = evaluate_blend(test_set, ruleset, symbolic_model, neural_model, device, blend_names)
    print(f"\n[Final Results] Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()