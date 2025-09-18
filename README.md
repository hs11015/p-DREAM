# p-DREAM (Prompt-Driven Reprogramming Exploiting Audio Mapping)

Diagnosing depression is critical due to its profound impact on individuals and associated risks.
Although deep learning techniques like convolutional neural networks and transformers have been employed to detect depression, they require large, labeled datasets and substantial computational resources, posing challenges in data-scarce environments.
We introduce p-DREAM (Prompt-Driven Reprogramming Exploiting Audio Mapping), a novel and data-efficient model designed to diagnose depression from speech data alone.
The p-DREAM combines two main strategies: data augmentation and model reprogramming. First, it utilizes audio-specific data augmentation techniques to generate a richer set of training examples.
Next, it employs audio prompts to aid in domain adaptation. These prompts guide a frozen pre-trained transformer, which extracts meaningful features. Finally, these features are fed into a lightweight classifier for prediction.
The p-DREAM outperforms traditional fine-tuning and linear probing methods, while requiring only a small number of trainable parameters. Evaluations on three benchmark datasets (DAIC-WoZ, E-DAIC, and AVEC 2014) demonstrate consistent improvements.
In particular, p-DREAM achieves a leading macro F1 score of 0.7734 using only acoustic features.
We further conducted ablation studies on prompt length, position, and initialization, confirming their importance in effective model adaptation.
p-DREAM offers a practical and privacy-conscious approach for speech-based depression assessment in low-resource environments.
To promote reproducibility and community adoption, now we plan to release our codebase in compliance with the ethical protocols outlined in the AVEC challenges.

## Our Methods

<img width="1723" height="592" alt="Figure1_ver3" src="https://github.com/user-attachments/assets/3808131b-efc1-469d-9f0d-f5550caa6b48" />

[The overview of our methods: p-DREAM]

### Augmentation Scheme

<img width="500" height="200" alt="audio_augmentation" src="https://github.com/user-attachments/assets/d1d5320d-09ce-48cc-afc9-ce9485fb65fa" />

[Augmentation Scheme overview]

<img width="900" height="300" alt="Figure2_ver4" src="https://github.com/user-attachments/assets/96cb56ec-17cc-4d8c-8f86-dfdcd9ca6951" />

[Example of a real augmentation on a real participant]

### Prompt Concatenation

<img width="650" height="450" alt="prompt_2" src="https://github.com/user-attachments/assets/8c0a9eae-de3d-4bc3-91bc-0283637e6e60" />

[Example of audio prompt concatenation]

### Model Reprogramming

<img width="450" height="450" alt="model_architecture_2" src="https://github.com/user-attachments/assets/385b0885-36c5-4eeb-9d04-1ceeca4eed39" />

[Comparision between Fine-tuning, Linear probing, and Model Reprogramming (our method: p-DREAM)]
