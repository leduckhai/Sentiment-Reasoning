# Sentiment Reasoning for Healthcare

**<div align="center">ACL 2025 Industry Track (Oral)</div>**

<div align="center"><b>Khai-Nguyen Nguyen*</b>, <b>Khai Le-Duc*</b>, Bach Phan Tat, Duy Le, Long Vo-Dang, Truong-Son Hy</div>

<div align="center">*Equal contribution</div>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
  <img src="https://github.com/leduckhai/Sentiment-Reasoning/blob/master/sentiment_reasoning_pipeline.png" width="700"/>
</p>
<p align="center"><em>Sentiment Reasoning pipeline</em></p>

<p align="center">
  <img src="https://github.com/leduckhai/Sentiment-Reasoning/blob/master/SentimentReasoning_ACL2025.png" width="700"/>
</p>

* **Abstract:**
Transparency in AI healthcare decision-making is crucial. By incorporating rationales to explain reason for each predicted label, users could understand Large Language Models (LLMs)’s reasoning to make better decision. In this work, we introduce a new task - **Sentiment Reasoning** - for both speech and text modalities, and our proposed multimodal multitask framework and **the world's largest multimodal sentiment analysis dataset**. Sentiment Reasoning is an auxiliary task in sentiment analysis where the model predicts both the sentiment label and generates the rationale behind it based on the input transcript. Our study conducted on both human transcripts and Automatic Speech Recognition (ASR) transcripts shows that Sentiment Reasoning helps improve model transparency by providing rationale for model prediction with quality semantically comparable to humans while also improving model's classification performance (**+2% increase in both accuracy and macro-F1**)  via rationale-augmented fine-tuning. Also, no significant difference in the semantic quality of generated rationales between human and ASR transcripts. All code, data (**five languages - Vietnamese, English, Chinese, German, and French**) and models are published online.

* **Citation:**
Please cite this paper: [https://arxiv.org/abs/2407.21054](https://arxiv.org/abs/2407.21054)

``` bibtex
@misc{Sentiment_Reasoning,
      title={Sentiment Reasoning for Healthcare}, 
      author={Khai-Nguyen Nguyen and Khai Le-Duc and Bach Phan Tat and Duy Le and Long Vo-Dang and Truong-Son Hy},
      year={2024},
      eprint={2407.21054},
      url={https://arxiv.org/abs/2407.21054}, 
}
```

This repository contains scripts for automatic speech recognition (ASR) and sentiment reasoning using cascaded sequence-to-sequence (seq2seq) audio-language models. The provided scripts cover model preparation, training, inference, and evaluation processes, based on the dataset in the paper.

## Dataset and Pre-trained Models:
🤗 **HuggingFace Dataset**: [https://huggingface.co/datasets/leduckhai/Sentiment-Reasoning](https://huggingface.co/datasets/leduckhai/Sentiment-Reasoning)

🤗 **HuggingFace Models**: 

| Model Name       | Description                                | Link                                                                 |
|------------------|--------------------------------------------|----------------------------------------------------------------------|
| `Sentiment-Reasoning-Vi`     | LLM fine-tuned on Vietnamese set        | [Hugging Face models](https://huggingface.co/leduckhai/Sentiment-Reasoning/tree/main/Vietnamese_Vistral-7B) |
| `Sentiment-Reasoning-Eng`    | LLM fine-tuned on English set         | [Hugging Face models](https://huggingface.co/knguyennguyen/Sentiment-Reasoning-Eng) |

To load the model:
```
from transformers import AutoModel
model = AutoModel.from_pretrained("leduckhai/Sentiment-Reasoning", subfolder="Vietnamese_Vistral-7B", dtype="auto")
```

**Paperswithcodes** to be released soon!

<p align="center">
  <img src="https://github.com/leduckhai/Sentiment-Reasoning/blob/master/sentiment_reasoning_datasample.png" width="1000"/>
</p>
<p align="center"><em>Sample data format used in Sentiment Reasoning dataset</em></p>


## Contact

Core developers:

**Khai Le-Duc**
```
University of Toronto, Canada
Email: duckhai.le@mail.utoronto.ca
GitHub: https://github.com/leduckhai
```

**Khai-Nguyen Nguyen**
```
College of William and Mary, USA
GitHub: https://github.com/nkn002
Hugging Face: https://huggingface.co/knguyennguyen
```

