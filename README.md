# p-DREAM
Prompt-Driven Reprogramming Exploiting Audio Mapping (p-DREAM)

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
