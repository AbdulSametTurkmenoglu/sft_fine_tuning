# TinyLlama Supervised Fine-Tuning (SFT) with QLoRA

A production-ready framework for fine-tuning small language models using Hugging Face `transformers` and `peft` (QLoRA). This project focuses on instruction-tuning `TinyLlama-1.1B-Chat-v1.0` and similar models using the standard `transformers.Trainer` class as an alternative to `trl.SFTTrainer`.

##  Features

-  **QLoRA Integration**: Efficient 4-bit quantization training for GPU memory optimization
-  **Modular Architecture**: Clean separation of data processing, model configuration, and training logic
-  **Flexible Data Loading**: Support for both Hugging Face Hub datasets (`timdettmers/openassistant-guanaco`) and custom local `.jsonl` files
-  **CLI Support**: Full command-line interface with `argparse` for easy parameter customization
-  **Built-in Testing**: Includes inference scripts for post-training model evaluation
-  **Checkpoint Management**: Automatic model saving and LoRA adapter merging

##  Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ GPU memory (for TinyLlama with QLoRA)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AbdulSametTurkmenoglu/sft_fine_tuning.git
cd sft_fine_tuning
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: Ensure your PyTorch installation supports CUDA if you plan to use GPU acceleration.

##  Usage

### Training the Model

The main training script is `train.py`.

#### Example 1: Train with Custom Instructions (Default)

Uses the technical Q&A dataset from `data/custom_instructions.jsonl`:
```bash
python train.py
```

#### Example 2: Train with Guanaco Dataset from Hugging Face
```bash
python train.py --dataset guanaco --guanaco_samples 1000 --output_dir sft_guanaco_model
```

#### Example 3: Train a Different Model with Custom Epochs
```bash
python train.py --model_name "google/gemma-2b" --epochs 5 --output_dir sft_gemma_model
```

#### View All Available Arguments
```bash
python train.py --help
```

### Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Base model to fine-tune |
| `--dataset` | `custom` | Dataset choice: `custom` or `guanaco` |
| `--custom_data_path` | `data/custom_instructions.jsonl` | Path to custom JSONL file |
| `--guanaco_samples` | `1000` | Number of Guanaco samples to use |
| `--output_dir` | `sft_tinyllama_model` | Directory to save the model |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Training batch size |
| `--learning_rate` | `2e-4` | Learning rate |

### Inference with Fine-Tuned Model

After training completes, use `inference.py` to interact with your model:
```bash
python inference.py --model_path sft_tinyllama_model
```

For models trained with different configurations:
```bash
python inference.py --base_model "google/gemma-2b" --model_path "sft_gemma_model"
```

#### Inference Features

- Automatically merges LoRA adapters with base model using `merge_and_unload()`
- Interactive chat interface
- Optimized for fast generation
- Supports custom system prompts

##  Project Structure
```
sft_fine_tuning/
├── train.py                    # Main training script
├── inference.py                # Inference and testing script
├── data/
│   └── custom_instructions.jsonl  # Custom training data
├── sft_tinyllama_model/        # Output directory (created after training)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

##  Data Format

### Custom JSONL Format

Your custom data should be in JSONL format with the following structure:
```json
{"instruction": "What is machine learning?", "output": "Machine learning is..."}
{"instruction": "Explain neural networks", "output": "Neural networks are..."}
```

Each line should contain:
- `instruction`: The user's question or prompt
- `output`: The expected model response

### Guanaco Dataset Format

When using `--dataset guanaco`, the script automatically loads and formats the OpenAssistant Guanaco dataset.

##  Configuration

### QLoRA Settings

The project uses 4-bit quantization with the following configuration:
```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
```

### LoRA Configuration
```python
lora_config = LoraConfig(
    r=16,                        # LoRA rank
    lora_alpha=32,              # Scaling factor
    target_modules=["q_proj", "v_proj"],  # Target attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### Training Hyperparameters

Default training configuration:

- **Learning Rate**: 2e-4
- **Batch Size**: 4 (per device)
- **Gradient Accumulation**: 4 steps
- **Max Steps**: 100
- **Warmup Steps**: 10
- **Logging**: Every 5 steps
- **Save Strategy**: At the end of training

##  Tips for Best Results

1. **Dataset Quality**: Ensure your instruction data is clean and well-formatted
2. **Learning Rate**: Start with 2e-4 and adjust based on loss curves
3. **Epochs**: 3-5 epochs typically work well for instruction tuning
4. **Evaluation**: Regularly test your model during training with sample prompts
5. **GPU Memory**: If you encounter OOM errors, reduce batch size or use gradient checkpointing

##  Troubleshooting

### CUDA Out of Memory

- Reduce `--batch_size` (try 2 or 1)
- Enable gradient checkpointing in training args
- Use a smaller model

### Slow Training

- Ensure CUDA is available and being used
- Check GPU utilization with `nvidia-smi`
- Increase batch size if memory allows

### Poor Model Performance

- Increase training epochs
- Use more training data
- Adjust learning rate
- Try different LoRA rank values

##  Technical Details

### Model Architecture

- **Base Model**: TinyLlama-1.1B (default) or any compatible causal LM
- **Fine-Tuning Method**: QLoRA (Quantized Low-Rank Adaptation)
- **Quantization**: 4-bit NF4 with double quantization
- **Adapter**: Low-rank matrices on attention layers

### Training Process

1. Load base model with 4-bit quantization
2. Add LoRA adapters to target modules
3. Prepare instruction-response pairs
4. Train using Hugging Face Trainer
5. Save LoRA adapters
6. Merge adapters for inference

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


##  Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the training framework
- [PEFT](https://github.com/huggingface/peft) for QLoRA implementation
- [TinyLlama](https://github.com/jzhang38/TinyLlama) for the base model
- [OpenAssistant](https://github.com/LAION-AI/Open-Assistant) for the Guanaco dataset

##  Contact

Abdul Samet Türkmenoğlu - [GitHub Profile](https://github.com/AbdulSametTurkmenoglu)

Project Link: [https://github.com/AbdulSametTurkmenoglu/sft_fine_tuning](https://github.com/AbdulSametTurkmenoglu/sft_fine_tuning)

##  Useful Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)
