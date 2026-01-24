import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ========== ChatBot Class ==========
class ChatBot:
    """A local LLM-based chatbot for generating emotional support responses"""
    def __init__(self, model_path, system_prompt):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16
        ).to("cuda:2")  # 修改为你实际的显卡编号

        self.system_prompt = system_prompt
        self.history = []
        self.clear_history()

    def clear_history(self):
        """Clear conversation history and reset with system prompt"""
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self):
        """Generate a response using the local LLM"""
        prompt = self.tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,  # 响应建议不要太长，512足够
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

# ========== Main Processing Function ==========
def process_file(file_name, model_path, model_name, system_prompt):
    """Process JSON file and generate responses"""
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    bot = ChatBot(model_path, system_prompt)

    # 处理指定范围的数据 (1000 到 1300)
    target_data = data[1000:1300]

    for idx, conv_item in enumerate(target_data):
        conversations = conv_item.get("conversations", [])
        
        # 逐轮迭代
        for i in range(0, len(conversations) - 1, 2):
            user_turn = conversations[i]
            bot_turn = conversations[i + 1] if i + 1 < len(conversations) else None

            if user_turn["from"] != "human":
                continue

            bot.clear_history()
            
            # 构建历史背景
            for j, turn in enumerate(conversations[:i + 1]):
                if turn["from"] == "human":
                    content = turn["value"]
                    # 如果是当前这一轮，把 Strategy 和 State 喂给模型
                    if j == i:
                        strat = turn.get("pre_strategy", "None")
                        state = turn.get("state", "None")
                        # 格式化策略信息，确保模型能读到
                        content = f"[User State]: {state}\n[Recommended Strategy]: {strat}\n[User Utterance]: {turn['value']}"
                    
                    bot.history.append({"role": "user", "content": content})
                elif turn["from"] == "gpt":
                    bot.history.append({"role": "assistant", "content": turn["value"]})

            # 生成回复
            print(f"[{idx+1000}] Generating for: {user_turn['value'][:30]}...")
            try:
                response = bot.chat()
                print(f"Result: {response[:50]}...\n")
            except Exception as e:
                print(f"Error: {e}\n")
                response = "ERROR_GENERATION"

            # 按照变量名字写入内容
            if bot_turn:
                bot_turn[model_name] = response

    # 写回原文件（注意：由于我们切片了数据，如果是想保存全量数据，需谨慎处理）
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 完成！结果已保存至字段: {model_name}")

# ========== Execution Entry Point ==========
if __name__ == "__main__":
    # 配置区
    MODEL_NAME = "" 
    MODEL_PATH = "" + MODEL_NAME
    INPUT_FILE = "ESConv_test.json"
    
    SYSTEM_PROMPT = (
        "You are a professional emotional supporter. "
        "You will be provided with the user's emotional state and a recommended support strategy. "
        "Please generate a response that strictly follows the strategy, "
        "maintaining a warm, empathetic, and concise tone."
    )

    process_file(
        file_name=INPUT_FILE,
        model_path=MODEL_PATH, 
        model_name=MODEL_NAME, # 传递变量名
        system_prompt=SYSTEM_PROMPT
    )