import json
from openai import OpenAI
import time

# ========== ChatBot Class (replaced with API calls) ==========
class ChatBot:
    """A wrapper class for interacting with LLM APIs to generate responses"""
    def __init__(self, api_key, api_base, model_name, system_prompt=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.history = []
    
    def clear_history(self):
        """Clear conversation history and reset with system prompt"""
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def chat(self):
        """Generate a response using the LLM API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.9
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            time.sleep(5)
            return ""


# ========== Main Processing Function ==========
def process_file(file_name, api_key, api_base, model_name):
    """Process JSON file containing conversations and generate responses using LLM"""
    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Initialize ChatBot (using API)
    bot = ChatBot(api_key=api_key, 
                  api_base=api_base, 
                  model_name=model_name,
                  system_prompt="You are an emotional supporter. Based on the conversation history and the user's current state, use the given strategy to generate a brief, warm, and empathetic response. Keep it short and friendly."
                  )

    # Process specified range of data (indices 1000 to 1300)
    for idx, conv_item in enumerate(data[1000:1300]):
        conversations = conv_item.get("conversations", [])
        for i in range(0, len(conversations) - 1, 2):
            user_turn = conversations[i]
            bot_turn = conversations[i + 1] if i + 1 < len(conversations) else None

            # Skip non-human or non-bot turns
            if user_turn["from"] != "human" or (bot_turn and bot_turn["from"] != "gpt"):
                continue

            # Clear history before processing each turn
            bot.clear_history()
            
            # Build conversation history up to current user turn
            for j, turn in enumerate(conversations[:i + 1]):
                if turn["from"] == "human":
                    if j == i:
                        bot.history.append({
                            "role": "user",
                            "content": turn["value"],
                            "name": "user_with_extra"  # Optional field for additional context
                        })
                    else:
                        bot.history.append({
                            "role": "user",
                            "content": turn["value"]
                        })
                elif turn["from"] == "gpt":
                    bot.history.append({
                        "role": "assistant",
                        "content": turn["value"]
                    })

            # Generate response
            print(f"[{idx}] Generating for: {user_turn['value']}\n")
            try:
                response = bot.chat()
                print(f"Response: {response}\n")
            except Exception as e:
                print(f"Generation failed: {e}\n")
                response = ""

            # Store generated response in original data structure
            if bot_turn:
                bot_turn[f"{bot.model_name}"] = response  # Use model name as field identifier

    # Write back to file
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ All generations complete. Results saved to {file_name}")


# ========== Execution Entry Point ==========
if __name__ == "__main__":
    # ========== Configurable Parameters (all in one place for easy modification) ==========
    input_file = ""  # Input file path
    api_key = ""
    api_base = ""
    model_name = ""  # Available model
    # ====================================================

    process_file(
        file_name=input_file,
        api_key=api_key,
        api_base=api_base,
        model_name=model_name
    )
