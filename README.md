# Neuro-Sym Supporter

This repository contains the official implementation of the WWW 2026 submission **#rfp1651**:

> **Neuro-Sym Supporter: A Thoughtful Emotion Support Agent Integrating Neural and Symbolic Policy Learning**

The project proposes a hybrid agent that integrates **neural strategy prediction** and **symbolic rule-based reasoning** for empathetic dialogue strategy selection.

---

## 📁 Repository Structure

```text
Neuro-Sym_Supporter/
├── data/
│   ├── empathicdialogues/
│   │   ├── ED_train.json
│   │   ├── ED_valid.json
│   │   └── ED_test.json
│   ├── ESConv/
│   │   ├── ESConv_train.json
│   │   ├── ESConv_valid.json
│   │   ├── ESConv_test.json
│   │   ├── ESConv_merged.json
│   │   └── ESConv_with_symbolic_state.json
│   ├── ruleset/
│   │   └── ruleset.json
│   ├── split_data.py
│   ├── extract_rules.py
│   └── evaluate_confidence.py
│
├── models/
│   ├── neural_mind.pth
│   └── best_neural_strategy_predictor.pth
│
├── Neuro-Sym-Mind/
│   ├── train_neuro_mind.py
│   ├── train_neuro_sym_mind.py
│   ├── train_sym_mind.py
│   ├── predict_nero_strategy.py
│   ├── predict_sym_strategy.py
│   ├── predict_blending_strategy.py
│   ├── split_data.py
│   └── check_data.py
│
├── LLaMA-Factory/
│   ├── src/
│   │   └──train.py
│   ├── generate_text.py
│   └── evaluate_result.py
│
└── README.md
```

---

## ⚙️ Environment Setup

We recommend using **Python ≥ 3.9** and managing dependencies via **conda** or **virtualenv**.

```bash
conda create -n nss python=3.10
conda activate nss
```

(Dependency installation follows the configuration of `LLaMA-Factory` and standard PyTorch setups.)

---

## 🚀 Running Instructions

### 1️⃣ Symbolic Rule Extraction

This step extracts symbolic dialogue strategies and confidence-aware rules from the ESConv dataset.

```bash
cd data
python evaluate_confidence.py
python extract_rules.py
```

Generated rules will be saved under `data/ruleset/`.

---

### 2️⃣ Neuro-Mind (Neural Strategy Predictor)

Train the neural-only strategy prediction model:

```bash
cd Neuro-Sym-Mind
python train_neuro_mind.py
```

Inference:

```bash
python predict_nero_strategy.py
```

---

### 3️⃣ Sym-Mind (Symbolic Strategy Reasoner)

Train the symbolic reasoning module based on extracted rules:

```bash
cd Neuro-Sym-Mind
python train_sym_mind.py
```

Inference:

```bash
python predict_sym_strategy.py
```

---

### 4️⃣ Neuro-Sym Blend (Hybrid Decision Module)

Train the blended policy that dynamically combines neural and symbolic strategies:

```bash
cd Neuro-Sym-Mind
python train_neuro_sym_mind.py
```

Inference:

```bash
python predict_blending_strategy.py
```

---

### 5️⃣ Supervised Fine-Tuning (SFT)

We adopt **LLaMA-Factory** for supervised fine-tuning of the dialogue model:

```bash
cd LLaMA-Factory
python src/train.py
<<<<<<< HEAD
python generate_text.py
python evaluate_result.py
=======
>>>>>>> 57aa4abd265da6de89662b277c3ff48a95e07eb1
```

Please refer to the official LLaMA-Factory documentation for dataset and configuration details.

---

## 📌 Notes

- All datasets are preprocessed JSON files aligned with ESConv and EmpatheticDialogues.
- Model checkpoints are saved under the `models/` directory.
- Prediction scripts support batch inference for evaluation and analysis.

---

## 📄 Citation

<<<<<<< HEAD
If you find this work useful, please cite our paper.
=======
If you find this work useful, please cite our WWW 2026 paper.
>>>>>>> 57aa4abd265da6de89662b277c3ff48a95e07eb1

```bibtex

```

