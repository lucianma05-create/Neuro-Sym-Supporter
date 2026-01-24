import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import logging
import sys
import random
from collections import deque
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Tuple
from tqdm import tqdm

# 引入划分模块
from splite_data import prepare_and_load_datasets

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ====== 模型组件 (保持不变) ======
class NeuralStateExtractor(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
    def forward(self, input_text: str) -> torch.Tensor:
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]

class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# ====== 核心预测器 ======
class NeuralStrategyPredictor(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_size=256, learning_rate=1e-4):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_extractor = NeuralStateExtractor(bert_model_name).to(self.device)
        bert_out = self.state_extractor.bert.config.hidden_size
        
        self.policy_net = DQNNetwork(bert_out, hidden_size, 8).to(self.device)
        self.target_net = DQNNetwork(bert_out, hidden_size, 8).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # a网络：用于Neuro-Sym混合时的动态权重
        self.a_net = nn.Sequential(
            nn.Linear(bert_out, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.a_net.parameters()), lr=learning_rate)
        self.memory = ReplayBuffer(10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        
        self.strategy_to_idx = {
            "Question": 0, "Affirmation and Reassurance": 1, "Information": 2,
            "Self-disclosure": 3, "Providing Suggestions": 4, "Restatement or Paraphrasing": 5,
            "Others": 6, "Reflection of feelings": 7
        }
        self.idx_to_strategy = {v: k for k, v in self.strategy_to_idx.items()}
        self.correct_predictions = 0
        self.total_predictions = 0

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(8)
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].item()

    def optimize_model(self, batch_size=32):
        if len(self.memory) < batch_size: return
        transitions = self.memory.sample(batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], device=self.device)
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device, dtype=torch.float32)

        current_q = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self, dialogue_turns):
        total_reward = 0
        correct = 0
        preds = 0
        
        for i in range(len(dialogue_turns) - 1):
            speaker, content, strategies = dialogue_turns[i]
            if speaker == "seeker":
                state = self.state_extractor(content)
                action = self.select_action(state)
                pred_str = self.idx_to_strategy[action]
                
                # 下一轮如果是supporter，检查策略
                actual_str = dialogue_turns[i+1][2] if dialogue_turns[i+1][0] == "supporter" else []
                reward = 1.0 if pred_str in actual_str else 0.0
                
                # 统计
                if reward > 0: 
                    correct += 1
                    self.correct_predictions += 1
                preds += 1
                self.total_predictions += 1
                total_reward += reward

                # 简单处理Next State
                next_state = state # 简化版
                self.memory.push(state, action, reward, next_state, (i == len(dialogue_turns)-2))
                self.optimize_model()
        
        self.epsilon = max(0.1, self.epsilon * 0.995)
        return total_reward, (correct/preds if preds > 0 else 0)

    def evaluate(self, dialogue_turns):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(len(dialogue_turns) - 1):
                if dialogue_turns[i][0] == "seeker" and dialogue_turns[i+1][0] == "supporter":
                    state = self.state_extractor(dialogue_turns[i][1])
                    action = self.select_action(state, training=False)
                    if self.idx_to_strategy[action] in dialogue_turns[i+1][2]:
                        correct += 1
                    total += 1
        return correct, total

    def save_model(self, path):
        torch.save(self.state_dict(), path)

# ====== 数据预处理工具 ======
def merge_turns(dialogue):
    merged = []
    last_speaker, last_utter, last_strategy = None, [], []
    for turn in dialogue:
        speaker = turn['speaker']
        content = turn.get('content', '')
        strategy = turn.get('annotation', {}).get('strategy', [])
        if speaker == last_speaker:
            last_utter.append(content)
            if speaker == 'supporter': last_strategy.extend(strategy)
        else:
            if last_speaker: merged.append((last_speaker, ' '.join(last_utter), list(set(last_strategy))))
            last_speaker, last_utter, last_strategy = speaker, [content], list(set(strategy))
    if last_speaker: merged.append((last_speaker, ' '.join(last_utter), list(set(last_strategy))))
    return merged

# ====== 主训练逻辑 ======
def main():
    DATA_PATH = "../data/ESConv/ESConv_merged.json" # 使用合并后的大数据集
    logger.info(f"Loading and splitting data from {DATA_PATH}...")
    train_set, valid_set, test_set = prepare_and_load_datasets(DATA_PATH)

    predictor = NeuralStrategyPredictor()
    num_episodes = 5
    best_val_acc = 0

    for ep in range(num_episodes):
        predictor.train()
        train_accs = []
        # 训练集
        for item in tqdm(train_set, desc=f"Epoch {ep+1} Training"):
            merged = merge_turns(item['dialog'])
            _, acc = predictor.train_episode(merged)
            train_accs.append(acc)
        
        # 验证集
        predictor.eval()
        val_correct, val_total = 0, 0
        for item in valid_set:
            merged = merge_turns(item['dialog'])
            c, t = predictor.evaluate(merged)
            val_correct += c
            val_total += t
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        logger.info(f"Epoch {ep+1} | Train Acc: {np.mean(train_accs):.4f} | Val Acc: {val_acc:.4f} | Eps: {predictor.epsilon:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            predictor.save_model("../models/neuro_mind.pth")
            logger.info("Saved new best model.")

        # 定期更新 Target Net
        predictor.target_net.load_state_dict(predictor.policy_net.state_dict())

# ====== 评估模式 ======
def evaluate_only(data_path="../data/ESConv/ESConv_merged.json"):
    _, _, test_set = prepare_and_load_datasets(data_path)
    predictor = NeuralStrategyPredictor()
    predictor.load_state_dict(torch.load("../models/neuro_mind.pth"))
    predictor.eval()
    
    logger.info(f"Evaluating on Test Set ({len(test_set)} dialogues)...")
    correct, total = 0, 0
    for item in tqdm(test_set):
        merged = merge_turns(item['dialog'])
        c, t = predictor.evaluate(merged)
        correct += c
        total += t
    
    print(f"\nFinal Test Accuracy: {correct/total:.4f} ({correct}/{total})")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate_only()
    else:
        main()