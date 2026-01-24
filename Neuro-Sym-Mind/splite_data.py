import json
import random
import os

def prepare_and_load_datasets(input_path, train_size=1040, valid_size=130, test_size=130, seed=42):
    """
    读取原始数据，进行打乱并划分为 训练/验证/测试集。
    返回: train_data, valid_data, test_data
    """
    # 1. 读取原始数据
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到原始数据文件: {input_path}")
        
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. 校验数据量
    total_expected = train_size + valid_size + test_size
    if len(data) < total_expected:
        print(f"⚠️ 警告：实际数据量({len(data)})少于期望总量({total_expected})，将按比例缩减。")
        # 如果数据不够，可以按比例重新计算 size，或者直接报错
    
    # 3. 打乱数据
    random.seed(seed)
    random.shuffle(data)

    # 4. 执行划分
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size : total_expected]

    print(f"✅ 数据集划分完成 (Seed: {seed})：")
    print(f"   Train: {len(train_data)} | Valid: {len(valid_data)} | Test: {len(test_data)}")
    
    return train_data, valid_data, test_data