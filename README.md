# Mindora

<div align="left">
    <a target='_blank'>Mengxi Xiao<sup>1</sup></span>&emsp;
    <a target='_blank'>Kailai Yang<sup>2</sup></span>&emsp;
    <a target='_blank'>Pengde Zhao<sup>3</sup></span>&emsp;
    <a target='_blank'>Enze Zhang<sup>1</sup></a>&emsp;
    <a target='_blank'>Ziyan Kuang<sup>4</sup></a>&emsp;
    <a target='_blank'>Zhiwei Liu<sup>2</sup></a>&emsp;
    <a target='_blank'>Weiguang Han<sup>1</sup></a>&emsp;
    <a target='_blank'>Min Peng<sup>1</sup></a>&emsp;
    <a target='_blank'>Qianqian Xie<sup>1*</sup></a>&emsp;
    <a target='_blank'>Sophia Ananiadou<sup>2</sup></a>
</div>

<br />
<div align="left">
    <sup>1</sup>School of Artificial Intelligence, Wuhan University&emsp;
    <sup>2</sup>The University of Manchester&emsp;
    <sup>3</sup>School of Computer Science, Wuhan University&emsp;
    <sup>4</sup>Jiangxi Normal University
</div>
<br />

Psychological Reasoning LLMs.

\[ English | [简体中文](README_zh.md) \]

## Latest News

✨[2025.9.4] We released the first model of our family, Mindora-rl2. For downloading the model checkpoints, please click here: [elsashaw/mindora-rl2](https://www.modelscope.cn/models/elsashaw/mindora-rl2/summary)

## Introduction

Mindora is a family of psychological reasoning LLMs designed for psychology-related tasks that demand strong reasoning abilities, including question answering, therapy plan generation, cognitive error analysis, and misinformation detection. We further evaluated the generalization ability of Mindora on unseen tasks such as psychiatric diagnosis and observed remarkable results.

Our base model is [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B), and we obtained it through SFT and GRPO.

## Results

|             | MHQA   | MedQA_en | MedMCQA | PubMedQA | AnnoMI | Therapist Q&A | DepSign | swmh   | t-sid  | misinfo |
| ----------- | ------ | -------- | ------- | -------- | ------ | ------------- | ------- | ------ | ------ | ------- |
| GPT-4o      | 0.4531 | 0.6138   | 0.7936  | 0.6949   | 0.3358 | 0.4543        | 0.4064  | 0.75   | 0.7114 | 0.6894  |
| Deepseek-r1 | 0.4984 | 0.8608   | 0.8988  | 0.7256   | 0.2020 | 0.4472        | 0.4085  | 0.7624 | 0.7218 | 0.5689  |
| Qwen3-8b    | 0.4009 | 0.5824   | 0.7927  | 0.6899   | 0.1489 | 0.4057        | 0.6395  | 0.7358 | 0.7515 | 0.4773  |
| Mindora-rl2 | 0.5996 | 0.7839   | 0.8593  | 0.8212   | 0.3803 | 0.4566        | 0.524   | 0.7715 | 0.814  | 0.5681  |



## Quick Start

You can use the model in the same way as using Qwen3-8B.

-   Initialization

```python
from modelscope import AutoModelForCausalLM, AutoTokenizer


class Mindora:
    def __init__(self, version):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name="elsashaw/mindora-rl2",
            torch_dtype="auto",
            device_map="cuda"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate(self, prompt):
        messages = [
            {"role": "system", "content": "Based on the following information of a case to make judgements. When answering, follow these steps concisely:\n\n 1. Reasoning Phase:\n   - Enclose all analysis within <think> tags\n   - Use structured subtitles (e.g., '###Comparing with Given Choices:') on separate lines\n   - Final section must be '###Final Conclusion:'\n\n2. Answer Phase:\n - Enclose your answer within <answer> tags\n - The answer phase should end with 'Answer: [option]'.\n - The answer should be aligned with reasoning phase. \nDeviation from this format is prohibited."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
```
-   Usage example

```python
mindora = Mindora()
response = mindora.generate(prompt="your prompt")
```


## Acknowledgments
Model training is based on the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [VeRL](https://github.com/volcengine/verl) frameworks.
