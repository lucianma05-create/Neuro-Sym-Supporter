import json

def diagnostic_data(file_path):
    print(f"🔍 正在检查文件: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return

    missing_count = 0
    total_seeker_turns = 0
    
    for d_idx, item in enumerate(data):
        # 这里的 'dialog' 是 ESConv 原始数据的标准字段
        dialog = item.get("dialog", [])
        for t_idx, turn in enumerate(dialog):
            if turn.get("speaker") == "seeker":
                total_seeker_turns += 1
                annotation = turn.get("annotation", {})
                
                # 核心检查点
                if "symbolic_state" not in annotation:
                    missing_count += 1
                    # 打印具体位置：第几个对话，第几个轮次，以及内容片段
                    content_snippet = turn.get("content", "").strip()[:30]
                    print(f"⚠️  缺失打分 -> 对话索引: {d_idx}, 轮次索引: {t_idx}")
                    print(f"   内容内容: \"{content_snippet}...\"")
                    print(f"   Annotation内容: {annotation}")
                    print("-" * 30)

    print("\n" + "="*30)
    print(f"📊 检查报告:")
    print(f"   - Seeker 总轮次: {total_seeker_turns}")
    print(f"   - 缺失 symbolic_state 的轮次: {missing_count}")
    
    if missing_count == 0:
        print("✅ 该文件格式完美，可以直接用于训练！")
    else:
        print(f"❌ 发现 {missing_count} 处异常。请检查打分脚本是否完整运行。")

if __name__ == "__main__":
    # 请确保路径与你训练脚本中的路径一致
    FILE_TO_CHECK = '../data/ESConv/ESConv_with_symbolic_state.json'
    diagnostic_data(FILE_TO_CHECK)