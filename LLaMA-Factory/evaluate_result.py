import json
from nltk.translate import meteor_score
from rouge_score import rouge_scorer
from statistics import mean
import sacrebleu
from nltk.tokenize import WordPunctTokenizer
from nltk.util import ngrams
import sys

def evaluate_result(file_name, value_name):
    """
    评估生成结果的质量，包括 ROUGE-L、METEOR、BLEU-1/2/3/4 等指标。
    评估文件中的所有对话。
    """
    tokenizer = WordPunctTokenizer()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute_corpus_sacrebleu(references: list, candidates: list) -> float:
        if not candidates: return 0.0
        # sacrebleu 的 corpus_bleu 要求 refs 是 [[ref1, ref2, ...]]
        return sacrebleu.corpus_bleu(candidates, [references]).score / 100.0

    def compute_rouge_l(reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip(): return 0.0
        return scorer.score(reference, candidate)['rougeL'].fmeasure

    def compute_meteor(reference: str, candidate: str) -> float:
        if not reference.strip() or not candidate.strip(): return 0.0
        references = [tokenizer.tokenize(reference)]
        hypothesis = tokenizer.tokenize(candidate)
        return meteor_score.meteor_score(references, hypothesis)

    def compute_bleu_n(reference: str, candidate: str, n: int) -> float:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        if not reference.strip() or not candidate.strip(): return 0.0
        smoothie = SmoothingFunction().method1
        ref = [tokenizer.tokenize(reference)]
        cand = tokenizer.tokenize(candidate)
        weights = {
            1: (1, 0, 0, 0),
            2: (0.5, 0.5, 0, 0),
            3: (1/3, 1/3, 1/3, 0),
            4: (0.25, 0.25, 0.25, 0.25)
        }
        return sentence_bleu(ref, cand, weights=weights[n], smoothing_function=smoothie)

    def distinct_n(candidates, n):
        total_ngrams = 0
        unique_ngrams = set()
        for sent in candidates:
            tokens = tokenizer.tokenize(sent)
            ngs = list(ngrams(tokens, n)) if len(tokens) >= n else []
            total_ngrams += len(ngs)
            unique_ngrams.update(ngs)
        return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0

    # --- 读取数据 ---
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return

    references = []
    candidates = []

    b1_list, b2_list, b4_list = [], [], []
    rouge_l_list, meteor_list = [], []

    print(f"🚀 开始评估文件: {file_name}")
    print(f"📊 目标字段: {value_name}")

    # --- 移除切片，遍历整个 dataset ---
    for dialog in dataset:
        conversations = dialog.get("conversations", [])
        for turn in conversations:
            # 只有当角色是 gpt 且包含生成的 candidate 字段时才评估
            if turn.get("from") == "gpt" and value_name in turn:
                candidate = str(turn[value_name])
                reference = str(turn["value"])

                if not reference.strip(): continue # 跳过无参考答案的项

                references.append(reference)
                candidates.append(candidate)

                # 计算各项指标
                rouge_l_list.append(compute_rouge_l(reference, candidate))
                meteor_list.append(compute_meteor(reference, candidate))
                b1_list.append(compute_bleu_n(reference, candidate, 1))
                b2_list.append(compute_bleu_n(reference, candidate, 2))
                b4_list.append(compute_bleu_n(reference, candidate, 4))

    if not candidates:
        print("⚠️ 未发现可匹配的生成结果，请检查 value_name 是否正确。")
        return

    # 多样性指标
    dist1 = distinct_n(candidates, 1)
    dist2 = distinct_n(candidates, 2)
    dist3 = distinct_n(candidates, 3)

    # 汇总
    average_scores = {
        "Total Turns": len(candidates),
        "ROUGE-L (avg)": mean(rouge_l_list),
        "METEOR (avg)": mean(meteor_list),
        "BLEU-1 (avg)": mean(b1_list),
        "BLEU-2 (avg)": mean(b2_list),
        "BLEU-4 (avg)": mean(b4_list),
        "Dist-1": dist1,
        "Dist-2": dist2,
        "Dist-3": dist3
    }

    print("\n🎯 最终评估结果（全量数据）：")
    print("-" * 30)
    for metric, score in average_scores.items():
        if metric == "Total Turns":
            print(f"{metric}: {score}")
        else:
            print(f"{metric}: {score:.4f}")

if __name__ == "__main__":
    # ========== 配置区 ==========
    # 将此路径改为你的实际文件路径
    INPUT_FILE = "ESConv_test.json" 
    # 模型生成的字段名，例如 "Qwen2.5-14B-Instruct"
    TARGET_FIELD = "Qwen2.5-14B-Instruct" 
    # ===========================
    
    evaluate_result(INPUT_FILE, TARGET_FIELD)