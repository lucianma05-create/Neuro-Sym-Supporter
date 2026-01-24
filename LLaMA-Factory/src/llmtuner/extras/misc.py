import json
from typing import Dict, List, Any, Generator

def iterate_esconv_dataset(file_path: str) -> Generator[Dict[str, Any], None, None]:
    """
    遍历ESConv数据集的生成器函数
    
    Args:
        file_path (str): ESConv.json文件的路径
        
    Yields:
        Dict[str, Any]: 数据集中的每一项对话数据
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            yield item

def process_conversation(conversation: Dict[str, Any]) -> None:
    """
    处理单个对话的函数
    
    Args:
        conversation (Dict[str, Any]): 单个对话数据
    """
    # 这里可以添加具体的处理逻辑
    conversation_id = conversation.get('id', '')
    conversations = conversation.get('conversations', [])
    
    print(f"处理对话ID: {conversation_id}")
    for msg in conversations:
        role = msg.get('from', '')
        content = msg.get('value', '')
        print(f"{role}: {content}")

def main():
    # 使用示例
    file_path = "ESConv.json"
    for conversation in iterate_esconv_dataset(file_path):
        process_conversation(conversation)

if __name__ == "__main__":
    main() 