import json
import random
import os

# ===== 配置 =====
input_path = "./ESConv_SFT.json"
train_size = 1040
valid_size = 130
test_size = 130

# 输出文件名（当前目录）
train_file = "ESConv_train.json"
valid_file = "ESConv_valid.json"
test_file = "ESConv_test.json"

# ===== 读取数据 =====
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

assert len(data) == train_size + valid_size + test_size, \
    f"数据量不匹配：期望 {train_size + valid_size + test_size}，实际 {len(data)}"

# ===== 打乱（推荐，避免顺序偏置）=====
random.seed(42)  # 固定随机种子，保证可复现
random.shuffle(data)

# ===== 划分 =====
train_data = data[:train_size]
valid_data = data[train_size:train_size + valid_size]
test_data = data[train_size + valid_size:]

# ===== 写入文件 =====
with open(train_file, "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=2)

with open(valid_file, "w", encoding="utf-8") as f:
    json.dump(valid_data, f, ensure_ascii=False, indent=2)

with open(test_file, "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

print("✅ 数据集划分完成：")
print(f"  Train: {len(train_data)} → {train_file}")
print(f"  Valid: {len(valid_data)} → {valid_file}")
print(f"  Test : {len(test_data)}  → {test_file}")
