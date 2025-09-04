# Mindora

Psychological Reasoning LLMs.

\[ English | [简体中文](README_zh.md) \]

## Latest News

✨[2025.9.4] We released the first model of our family, Mindora-rl2. For model downloads, please click here: [elsashaw/mindora-rl2](https://www.modelscope.cn/models/elsashaw/mindora-rl2/summary)

## Introduction

Mindora is a family of psychological reasoning LLMs designed for psychology-related tasks that demand strong reasoning abilities, including question answering, therapy plan generation, cognitive error analysis, and misinformation detection. We further evaluated the generalization ability of Mindora on unseen tasks such as psychiatric diagnosis and observed remarkable results.

Our base model is [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B), and we obtained it through SFT and GRPO.



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
