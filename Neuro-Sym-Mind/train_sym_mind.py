import json
import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
from tqdm import tqdm
# 导入你的划分函数
from splite_data import prepare_and_load_datasets

# ====== 常用函数 ======
def soft_sigmoid(x, k=10):
    """平滑归一化函数"""
    return 1 / (1 + torch.exp(-k * (x - 0.5)))

def geometric_mean_torch(vals):
    """使用 Tensor 实现几何均值，保持梯度链"""
    if vals.numel() == 0:
        return torch.tensor(0.0, device=vals.device)
    return torch.exp(torch.mean(torch.log(vals + 1e-8)))

# ====== 策略预测模型定义 ======
class StrategyPredictor(nn.Module):
    def __init__(self, num_rules, initial_k=10.0):
        super(StrategyPredictor, self).__init__()
        self.rule_weights = nn.Parameter(torch.ones(num_rules))
        self.k = nn.Parameter(torch.tensor(initial_k))
    
    def soft_sigmoid(self, x):
        return 1 / (1 + torch.exp(-self.k * (x - 0.5)))
    
    def forward(self, rule_scores):
        weighted_scores = rule_scores * self.rule_weights
        return weighted_scores

# ====== GPU/CPU 自动切换 ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== 加载模型参数 ======
def load_model(ruleset, model_path='../models/strategy_model.pt'):
    num_rules = sum(len(rules) for rules in ruleset.values())
    model = StrategyPredictor(num_rules=num_rules).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

# ====== 规则匹配与策略预测 ======
def matching_roles(seeker_data, ruleset, model):
    """单样本推理函数"""
    annotation = seeker_data["annotation"]
    raw_text_state = annotation["symbolic_state"]
    
    all_rules = []
    strategy_rule_indices = {}
    rule_idx = 0
    for strategy, rules in ruleset.items():
        indices = []
        for rule in rules:
            all_rules.append(rule)
            indices.append(rule_idx)
            rule_idx += 1
        strategy_rule_indices[strategy] = indices

    rule_scores_list = []
    for rule in all_rules:
        rule_keys = rule["symbolic_state"]
        values = []
        for key in rule_keys:
            v = raw_text_state.get(key, 0.0)
            values.append(model.soft_sigmoid(torch.tensor(v, dtype=torch.float32, device=device)))
        
        # 计算几何均值，已删除 current_nodetype 惩罚逻辑
        if values:
            gm = geometric_mean_torch(torch.stack(values))
        else:
            gm = torch.tensor(0.0, device=device)
        rule_scores_list.append(gm)

    rule_scores_tensor = torch.stack(rule_scores_list)
    weighted_rule_scores = model(rule_scores_tensor)

    strategy_scores = []
    strategy_names = list(strategy_rule_indices.keys())
    for strategy in strategy_names:
        indices = strategy_rule_indices[strategy]
        strategy_scores.append(torch.sum(weighted_rule_scores[indices]))

    strategy_scores_tensor = torch.stack(strategy_scores)
    norm_probs = torch.softmax(strategy_scores_tensor, dim=0)
    
    pre_idx = torch.argmax(norm_probs).item()
    return strategy_names[pre_idx], norm_probs, strategy_names

# ====== 批量并行规则得分计算 ======
def batch_compute_rule_scores(batch_dialogs, all_rules, model, device):
    B = len(batch_dialogs)
    num_rules = len(all_rules)
    rule_scores = torch.zeros((B, num_rules), dtype=torch.float32, device=device)
    
    for i, seeker_dialog in enumerate(batch_dialogs):
        raw_text_state = seeker_dialog["annotation"]["symbolic_state"]
        
        for j, rule in enumerate(all_rules):
            rule_keys = rule["symbolic_state"]
            values = [model.soft_sigmoid(torch.tensor(raw_text_state.get(k, 0.0), dtype=torch.float32, device=device)) for k in rule_keys]
            
            if values:
                gm = geometric_mean_torch(torch.stack(values))
            else:
                gm = torch.tensor(0.0, device=device)
            # 已删除惩罚逻辑
            rule_scores[i, j] = gm
    return rule_scores

# ====== 训练样本准备 ======
def prepare_training_samples(data, ruleset):
    samples = []
    strategy_names = list(ruleset.keys())
    for item in data:
        dialog = item["dialog"]
        for i in range(len(dialog) - 1):
            if dialog[i]["speaker"] == "seeker" and dialog[i + 1]["speaker"] == "supporter":
                real_strategies = dialog[i + 1]["annotation"].get("strategy", [])
                if real_strategies and real_strategies[0] in strategy_names:
                    samples.append((dialog[i], real_strategies[0]))
    return samples, strategy_names

# ====== 评估函数 ======
def evaluate(model, eval_data, ruleset):
    model.eval()
    real_example = 0
    example = 0
    with torch.no_grad():
        for item in eval_data:
            dialog = item["dialog"]
            for i in range(len(dialog) - 1):
                if dialog[i]["speaker"] == "seeker" and dialog[i+1]["speaker"] == "supporter":
                    pre_strategy, _, _ = matching_roles(dialog[i], ruleset, model)
                    real_strategy_list = dialog[i+1]["annotation"].get("strategy", [])
                    if pre_strategy in real_strategy_list:
                        real_example += 1
                    example += 1
    
    accuracy = real_example / example if example > 0 else 0.0
    return accuracy

# ====== 训练主函数 ======
def train(train_data, test_data, ruleset, save_path='../models/sym_mind.pt', batch_size=64):
    num_rules = sum(len(rules) for rules in ruleset.values())
    model = StrategyPredictor(num_rules=num_rules).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 3
    
    samples, strategy_names = prepare_training_samples(train_data, ruleset)
    all_rules = []
    strategy_rule_indices = {}
    rule_idx = 0
    for strategy, rules in ruleset.items():
        indices = []
        for rule in rules:
            all_rules.append(rule); indices.append(rule_idx); rule_idx += 1
        strategy_rule_indices[strategy] = indices

    num_batches = (len(samples) + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        random.shuffle(samples)
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx in pbar:
            batch = samples[batch_idx*batch_size : (batch_idx+1)*batch_size]
            batch_dialogs = [x[0] for x in batch]
            batch_targets = torch.tensor([strategy_names.index(x[1]) for x in batch], device=device)
            
            rule_scores = batch_compute_rule_scores(batch_dialogs, all_rules, model, device)
            weighted_scores = model(rule_scores)
            
            # 策略分组求和
            batch_logits = []
            for b in range(weighted_scores.shape[0]):
                s_scores = torch.stack([torch.sum(weighted_scores[b][strategy_rule_indices[s]]) for s in strategy_names])
                batch_logits.append(s_scores)
            logits_tensor = torch.stack(batch_logits)
            
            loss = loss_fn(logits_tensor, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(samples)
        acc = evaluate(model, test_data, ruleset)
        print(f"Epoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f} | Test Acc: {acc:.4f}")
        torch.save(model.state_dict(), f'../models/sym_mind_epoch{epoch+1}.pt')

    torch.save(model.state_dict(), save_path)
    print(f"Final Model saved to {save_path}")
    return model

# ====== 主程序入口 ======
if __name__ == '__main__':
    RAW_DATA_PATH = '../data/ESConv/ESConv_with_symbolic_state.json'
    RULESET_PATH = '../data/ruleset/ruleset.json'
    
    train_data, valid_data, test_data = prepare_and_load_datasets(RAW_DATA_PATH)
    
    with open(RULESET_PATH, 'r', encoding='utf-8') as f:
        ruleset = json.load(f)

    if len(sys.argv) > 1 and sys.argv[1] == 'eval':
        num_rules = sum(len(rules) for rules in ruleset.values())
        model = StrategyPredictor(num_rules=num_rules).to(device)
        model.load_state_dict(torch.load('../models/sym_mind.pt', map_location=device))
        print("Evaluating on Test Set...")
        acc = evaluate(model, test_data, ruleset)
        print(f"Final Test Accuracy: {acc:.4f}")
    else:
        print("Starting Symbolic Mind Training...")
        train(train_data, test_data, ruleset)