# DeepSeek-R1-Distill-Qwen-1.5B

A compact 1.5B parameter reasoning model distilled from DeepSeek-R1, built on the Qwen2.5-1.5B base model. This project provides tools and resources for running the DeepSeek-R1-Distill-Qwen-1.5B model locally.

## 🌟 Overview

DeepSeek-R1-Distill-Qwen-1.5B is part of the DeepSeek-R1-Distill model family - smaller, efficient models that inherit the reasoning capabilities of the larger DeepSeek-R1 model through knowledge distillation. Despite its compact size, this 1.5B parameter model demonstrates impressive performance on mathematical reasoning, coding, and general reasoning tasks.

### Key Features

- **Compact Size**: Only 1.5B parameters, suitable for resource-constrained environments
- **Strong Reasoning**: Inherits chain-of-thought reasoning capabilities from DeepSeek-R1
- **Versatile Performance**: Excels in mathematics, coding, and logical reasoning tasks
- **Open Source**: MIT licensed with commercial use support
- **Easy Integration**: Compatible with popular inference frameworks

## 📊 Performance Highlights

The model achieves remarkable performance for its size:

- **AIME 2024**: 28.9% pass@1 (outperforming GPT-4o's 9.3%)
- **MATH-500**: 83.9% pass@1 
- **GPQA Diamond**: 33.8% pass@1
- **CodeForces Rating**: 954
- **LiveCodeBench**: 16.9% pass@1

## 🚀 Quick Start

### Installation

```bash
# Install required dependencies
pip install torch transformers vllm accelerate

# Or for SGLang support
pip install sglang
```

### Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Generate response
prompt = "Please solve this step by step: What is 2^10 + 3^5?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    inputs.input_ids,
    max_length=1024,
    temperature=0.6,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using vLLM (Recommended for Production)

```bash
# Start vLLM server
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code
```

### Using SGLang

```bash
# Start SGLang server
python3 -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --trust-remote-code
```

## 🎯 Usage Recommendations

For optimal performance, follow these guidelines:

### Generation Parameters
- **Temperature**: 0.5-0.7 (0.6 recommended)
- **Top-p**: 0.95
- **Max tokens**: Up to 32,768
- **Multiple sampling**: Generate 64 responses for pass@1 evaluation

### Prompting Best Practices

1. **No System Prompts**: Include all instructions in the user prompt
2. **Mathematical Problems**: Add "Please reason step by step, and put your final answer within \\boxed{}."
3. **Force Reasoning**: Start responses with `<think>\n` to ensure step-by-step reasoning
4. **Clear Instructions**: Be specific about the desired output format

### Example Prompts

**Mathematical Reasoning:**
```
Please solve this step by step and put your final answer within \boxed{}: 
If a train travels 120 miles in 2 hours, what is its average speed in miles per hour?
```

**Coding Problems:**
```
Write a Python function to find the factorial of a number. Please explain your approach step by step.
```

**Logical Reasoning:**
```
<think>
Given the following premises, what can we conclude?
- All birds can fly
- Penguins are birds
- Penguins cannot fly
Please analyze this step by step.
```

## 🛠️ Technical Specifications

- **Base Model**: Qwen2.5-1.5B
- **Parameters**: 1.5 billion
- **Context Length**: 32,768 tokens
- **Training**: Fine-tuned with 800k samples from DeepSeek-R1
- **Architecture**: Transformer-based language model
- **Precision**: FP16/BF16 support

## 📦 Model Formats

The model is available in multiple formats:

- **HuggingFace**: Standard PyTorch format
- **GGUF**: Quantized format for efficient inference
- **ONNX**: Cross-platform deployment format

### Quantized Versions

For even more efficient deployment, consider using quantized versions:
- `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`
- `unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`

## 🔧 Integration Examples

### OpenAI-Compatible API

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # Your vLLM server
    api_key="token-abc123"  # Can be any string
)

response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[
        {"role": "user", "content": "Solve: 2x + 5 = 15"}
    ],
    temperature=0.6,
    max_tokens=1024
)

print(response.choices[0].message.content)
```

### Batch Processing

```python
def process_batch(questions, model, tokenizer):
    results = []
    for question in questions:
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(
            inputs.input_ids,
            max_length=1024,
            temperature=0.6,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append(response)
    return results
```

## 🎓 Use Cases

### Educational Applications
- Step-by-step problem solving
- Mathematical tutoring
- Code explanation and debugging
- Logical reasoning exercises

### Research Applications
- Reasoning capability analysis
- Model distillation experiments
- Benchmark evaluation
- Comparative studies

### Production Applications
- Automated problem solving
- Code generation assistance
- Mathematical computation
- Decision support systems

## 🔍 Evaluation and Benchmarking

### Running Evaluations

```python
# Example evaluation script
def evaluate_math_problems(model, tokenizer, problems):
    correct = 0
    total = len(problems)
    
    for problem in problems:
        prompt = f"Please solve step by step: {problem['question']}"
        # Generate response and check answer
        # Implementation details depend on your evaluation framework
        
    accuracy = correct / total
    return accuracy
```

### Benchmark Results

The model has been evaluated on various benchmarks:
- Mathematical reasoning (AIME, MATH-500)
- General knowledge (MMLU variants)
- Coding capabilities (LiveCodeBench, Codeforces)
- Scientific reasoning (GPQA Diamond)

## 📄 License

This project is licensed under the MIT License, allowing for:
- ✅ Commercial use
- ✅ Modification and derivative works
- ✅ Distribution
- ✅ Private use

**Note**: The base model (Qwen2.5-1.5B) is licensed under Apache 2.0 License.

## 🤝 Contributing

Contributions are welcome! Please consider:

1. **Bug Reports**: Open issues for any problems encountered
2. **Feature Requests**: Suggest improvements or new features
3. **Documentation**: Help improve documentation and examples
4. **Code**: Submit pull requests for bug fixes or enhancements

## 📚 Additional Resources

- **Paper**: [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- **HuggingFace Model**: [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- **Official Chat**: [chat.deepseek.com](https://chat.deepseek.com)
- **API Platform**: [platform.deepseek.com](https://platform.deepseek.com/)

## 📞 Support

For questions and support:
- 📧 Email: service@deepseek.com
- 🐛 Issues: Open a GitHub issue
- 💬 Community: Join the discussion on HuggingFace

## 🏷️ Citation

If you use this model in your research, please cite:

```bibtex
@misc{deepseekai2025deepseekr1incentivizingreasoningcapability,
    title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
    author={DeepSeek-AI},
    year={2025},
    eprint={2501.12948},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2501.12948},
}
```

---

**Note**: This model requires careful prompt engineering and parameter tuning to achieve optimal performance. Please refer to the usage recommendations section for best practices.
