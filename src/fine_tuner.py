import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings

warnings.filterwarnings('ignore')

class SFTFineTuner:
    """
    SFT kullanarak model ince ayarı yapan sınıf
    """

    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None

        print(f"SFT Fine-Tuning Başlatılıyor...")
        print(f" Tercih Edilen Cihaz: {self.device}")
        print(f" Model: {model_name}")
        print(f" Çıktı: {output_dir}")

    def setup_model_and_tokenizer(self, use_quantization: bool = True):
        """Model ve tokenizer'ı hazırla"""
        print("\nModel ve tokenizer yükleniyor...")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "right"

        if use_quantization and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print("   4-bit Quantization (QLoRA) etkin.")
        else:
            bnb_config = None
            if use_quantization and self.device != "cuda":
                print("   Uyarı: Quantization sadece CUDA ile desteklenir. Devre dışı bırakıldı.")

        print("   Model yükleniyor...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True,
        )

        actual_device = next(self.model.parameters()).device
        print(f"   Model yüklendi: {actual_device}")

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # TinyLlama için ortak
        )

        if use_quantization and self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        print(" Model (PEFT) hazır!")
        self.model.print_trainable_parameters()
        return self.model, self.tokenizer

    def create_sft_config(self, num_epochs: int = 3, batch_size: int = 4):
        print("\nSFT konfigürasyonu hazırlanıyor...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=2,
            optim="paged_adamw_8bit" if self.device == "cuda" else "adamw_torch",
            fp16=True if self.device == "cuda" else False,
            gradient_checkpointing=False,
            report_to="none",
            logging_first_step=True,
            load_best_model_at_end=True,
        )
        print(f" Konfigürasyon hazır!")
        return training_args

    def tokenize_dataset(self, dataset, max_length=512):
        """'text' sütununu kullanarak veri setini tokenize eder"""
        print(f"  Tokenize ediliyor (max_length={max_length})...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        return tokenized_dataset

    def train(self, dataset, training_args):
        """SFT eğitimini başlat (Standart 'transformers.Trainer' ile)"""
        print("\n SFT Eğitimi Başlıyor...")
        print("=" * 60)

        print(" 'transformers.Trainer' modu: 'trl' kullanılmıyor.")

        max_seq_len = 512
        print(f"  • Max sequence length: {max_seq_len}")
        tokenized_train_dataset = self.tokenize_dataset(dataset["train"], max_length=max_seq_len)
        tokenized_eval_dataset = self.tokenize_dataset(dataset["test"], max_length=max_seq_len)

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False # Causal LM için mlm=False olmalı
        )
        print("   DataCollatorForLanguageModeling (mlm=False) kullanılıyor.")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        print("\n Eğitim devam ediyor...")
        trainer.train()

        print("\n Eğitim tamamlandı!")

        print(f"\n Model kaydediliyor: {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print(" LoRA adaptörü ve tokenizer başarıyla kaydedildi.")
        return trainer