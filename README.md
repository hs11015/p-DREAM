# Prompt-Based Model Reprogramming for Efficient Depression Detection from Speech

> ICASSP 2026  
> **Efficient Depression Detection from Speech via Language-Independent Model Reprogramming**

---

## 🧠 Overview

Depression diagnosis from speech is a challenging task due to:
- limited labeled data,
- privacy constraints,
- and poor cross-dataset generalization.

Most deep learning approaches rely on **full fine-tuning**, which:
- requires large datasets,
- introduces heavy computational cost,
- and often overfits to dataset-specific biases.

---

### 🚀 Our Idea

We propose a **prompt-based model reprogramming framework**,  
a **parameter-efficient approach** that adapts pretrained speech models **without fine-tuning**.

> ❗ Instead of learning new parameters, we **reprogram the input space** using audio prompts.

---

## 🔑 Key Contributions

- **Prompt-based reprogramming for speech**
  - Treat prompts as **input-level perturbations**, not trainable modules  
- **Language-independent depression detection**
  - Uses only **acoustic features** (no ASR / text)  
- **Parameter-efficient adaptation**
  - more than 100,000× fewer trainable parameters than fine-tuning  
- **Strong performance under low-resource settings**  
- **Cross-dataset robustness**  
- **Systematic prompt design analysis (ablation study)**  

---

## 🏗️ Method Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/3808131b-efc1-469d-9f0d-f5550caa6b48" width="90%">
</p>

**Pipeline:**
1. Audio augmentation  
2. Prompt concatenation (**input-level reprogramming**)  
3. Frozen pretrained encoder (e.g., AST / HuBERT / Whisper / wav2vec2)  
4. Lightweight classifier  

> **Key insight:** Prompt acts as **input-level reprogramming**, not a trainable module.

---

### Augmentation Scheme

<img width="500" height="250" alt="audio_augmentation" src="https://github.com/user-attachments/assets/d1d5320d-09ce-48cc-afc9-ce9485fb65fa" />

[Augmentation Scheme overview]

<img width="900" height="300" alt="Figure2_ver4" src="https://github.com/user-attachments/assets/96cb56ec-17cc-4d8c-8f86-dfdcd9ca6951" />

[Example of a real augmentation on a real participant]

> Audio augmentation increases diversity and improves robustness under limited data.

---

### Prompt Concatenation

<img width="300" height="200" alt="prompt_2" src="https://github.com/user-attachments/assets/8c0a9eae-de3d-4bc3-91bc-0283637e6e60" />

[Example of audio prompt concatenation]

> Prompts are treated as **structured input signals** that steer the pretrained model.

---

### Model Reprogramming

<img width="450" height="450" alt="model_architecture_2" src="https://github.com/user-attachments/assets/385b0885-36c5-4eeb-9d04-1ceeca4eed39" />

[Comparison between Fine-tuning, Linear Probing, and Model Reprogramming (ours)]

> Unlike fine-tuning, our approach **does not modify backbone parameters**, enabling efficient and generalizable adaptation.

---

## 🎯 Core Concept: Input Reprogramming

Unlike conventional approaches:

| Method | Strategy |
|--------|----------|
| Fine-tuning | Update all parameters |
| Linear probing | Train classifier only |
| **Ours** | **Reprogram input distribution** |

> We **do not train prompts**.
> We use them to **steer the pretrained representation space**.

---

## 🎛️ Prompt Design

We systematically analyze how prompt design affects performance:

- Audio length  
- Prompt length  
- Initialization  
- Insertion position  

👉 Key finding:

> **Prompt design consistently affects model behavior → not a heuristic**

---

## 📊 Results

- Evaluated on:
  - DAIC-WoZ  
  - AVEC 2014  

- Achieves:
  - **Macro F1: 0.7734**  
  - using **acoustic-only input**

---

### 💡 Key Observations

- Moderate input length yields best performance  
- Prompt length has an optimal range  
- Initialization significantly affects sensitivity  
- **Simple prompts outperform complex designs**  

> ❗ Minimal perturbations are sufficient to steer pretrained models

---

## 🔬 Why It Works

We interpret prompts as:

> **structured input perturbations (reprogramming signals)**

Similar to:
- input filtering in vision (e.g., Gaussian filtering)  
- domain adaptation via input transformation  

This approach **shifts the representation space** without modifying model weights.

---

## 🌍 Generalization

We observe similar behavior in other modalities (e.g., EEG):

> Prompt-based reprogramming is **not modality-specific**
> PEARL: Prompt-Based EEG Adaptation via Resource-Efficient Learning for Generalizable Brain-Computer Interface (ICASSP 2026, accept)

---

## ⚡ Efficiency

- Minimal trainable parameters  
- No backbone updates  
- Faster inference than fine-tuning  

---

## 🧪 Ablation Study

We provide detailed analysis on:
- prompt length  
- audio duration  
- initialization strategy  
- insertion position  

👉 Demonstrates:

> prompt design is a **systematic factor**, not arbitrary tuning

---

## 🚧 Code Release

Due to **ethical and privacy constraints** (AVEC protocol),  
we are carefully preparing the public release of this repository.

> Planned release includes:
- Training and evaluation pipeline  
- Prompt generation and integration modules  
- Preprocessing and augmentation scripts  

⚠️ Note:  
We do **not redistribute raw audio data**.  
Users must obtain datasets (e.g., DAIC-WoZ, AVEC 2014) through official channels.

---

## ⚠️ Pretrained Model

Due to GitHub storage limitations, we do not provide pretrained weights directly in this repository.

Please download the pretrained model from the official source:

- **AudioSet pretrained checkpoint**: `audioset_10_10_0.4593.pth`

You can typically find this checkpoint via:
- the original model repository (e.g., PANNs / AudioSet pretrained models)
- or public model hosting platforms

After downloading, place the file in the following directory:

```bash
project/pretrained_models/audioset_10_10_0.4593.pth
```
---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{kim2026reprogramming,
  title={Efficient Depression Detection from Speech via Language-Independent Model Reprogramming},
  author={Hyunseo Kim, Longbin Jin, and Eun Yi Kim},
  booktitle={Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2026}
}
```
