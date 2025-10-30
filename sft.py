import torch
from datasets import load_dataset, Dataset
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
    (Eski kütüphanelerle uyumluluk için standart 'transformers.Trainer' kullanılır)
    """

    def __init__(
            self,
            model_name: str = "google/gemma-2b",
            dataset_name: str = "timdettmers/openassistant-guanaco",
            output_dir: str = "./sft_model"
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.tokenizer = None
        self.model = None

        print(f" SFT Fine-Tuning Başlatılıyor...")
        print(f" Tercih Edilen Cihaz: {self.device}")
        print(f" Model: {model_name}")
        print(f" Veri Seti: {dataset_name}")
        print(f"  Not: PyTorch modeli otomatik olarak en uygun cihaza taşıyabilir")

    def load_instruction_dataset(self, num_samples: int = 1000):
        """
        Veri setini yükle (standart 'text' sütununu kullanır)
        """
        print("\n Veri seti yükleniyor...")
        dataset = load_dataset(self.dataset_name, split="train")
        dataset = dataset.select(range(min(num_samples, len(dataset))))

        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        print(f" {len(dataset['train'])} eğitim, {len(dataset['test'])} test örneği yüklendi")
        if len(dataset['train']) > 0:
            print(f"\n Örnek veri (birleştirilmiş 'text' formatı):")
            print(dataset['train'][0]['text'][:300] + "...")
        return dataset

    def create_custom_dataset(self):
        """
        Özel talimat veri seti oluştur (tek 'text' sütunu olarak)
        """
        print("\n Özel veri seti oluşturuluyor...")

        if self.tokenizer is None:
            raise ValueError("HATA: create_custom_dataset çağrılmadan önce setup_model_and_tokenizer çağrılmalıdır.")

        instructions = [
            {"instruction": "Python'da liste comprehension nedir? Açıkla.",
             "response": "Liste comprehension, Python'da kısa ve okunabilir şekilde liste oluşturmanın bir yoludur. Örnek: squares = [x**2 for x in range(10)] şeklinde 0-9 arası sayıların karelerini içeren bir liste oluşturabilirsiniz."},
            {"instruction": "Makine öğrenmesinde overfitting nedir?",
             "response": "Overfitting, bir modelin eğitim verilerine aşırı uyum sağlayıp, yeni verilerde kötü performans göstermesi durumudur. Regularization, dropout, ve daha fazla veri ile önlenebilir."},
            {"instruction": "HTTP ve HTTPS arasındaki fark nedir?",
             "response": "HTTPS, HTTP'nin güvenli versiyonudur. SSL/TLS protokolü kullanarak veri şifrelemesi yapar. Web sitelerinde kilit simgesi görüyorsanız, HTTPS kullanılıyor demektir."},
            {"instruction": "Bir binary search tree nedir?",
             "response": "Binary Search Tree (BST), her düğümün en fazla 2 çocuğu olan ve sol alt ağacın değerlerinin kökten küçük, sağ alt ağacın değerlerinin ise büyük olduğu bir veri yapısıdır. Arama işlemleri O(log n) karmaşıklığındadır."},
            {"instruction": "Docker nedir ve neden kullanılır?",
             "response": "Docker, uygulamaları container'lar içinde paketleyip çalıştıran bir platformdur. Avantajları: taşınabilirlik, tutarlı çevre, kolay deployment, ve kaynak verimliliği. 'Works on my machine' sorununu çözer."},
            {"instruction": "Python'da decorator nedir?",
             "response": "Decorator, başka bir fonksiyonu saran ve davranışını değiştiren fonksiyondur. @decorator_name sözdizimi ile kullanılır. Loglama, yetkilendirme ve önbellekleme gibi durumlarda kullanışlıdır."},
            {"instruction": "REST API tasarlarken nelere dikkat etmeliyim?",
             "response": "REST API tasarlarken şunlara dikkat edin: HTTP metodlarını doğru kullanın (GET, POST, PUT, DELETE), anlamlı URL yapıları oluşturun, uygun status kodları döndürün, versiyonlama yapın ve güvenlik önlemleri alın (HTTPS, authentication)."},
            {"instruction": "Git merge ve rebase arasındaki fark nedir?",
             "response": "Git merge iki branch'i birleştirirken yeni bir commit oluşturur ve geçmişi korur. Rebase ise commit'leri yeniden yazar ve daha temiz bir geçmiş oluşturur. Merge güvenlidir, rebase geçmişi yeniden yazar."},
            {"instruction": "Makine öğrenmesinde regularization nedir?",
             "response": "Regularization, modelin overfitting yapmasını önleyen bir tekniktir. L1 (Lasso) ve L2 (Ridge) regularization gibi yöntemlerle model karmaşıklığı cezalandırılır. Bu sayede model yeni verilerde daha iyi performans gösterir."},
        ]

        formatted_data = []
        for item in instructions:
            text = f"### Human: {item['instruction']}\n### Assistant: {item['response']}"
            text += self.tokenizer.eos_token
            formatted_data.append({"text": text})

        dataset = Dataset.from_list(formatted_data)
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        print(f" {len(dataset['train'])} eğitim, {len(dataset['test'])} test örneği oluşturuldu")
        return dataset

    def setup_model_and_tokenizer(self, use_quantization: bool = True):
        """Model ve tokenizer'ı hazırla"""
        print("\n Model ve tokenizer yükleniyor...")

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
        else:
            bnb_config = None

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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )

        if use_quantization and self.device == "cuda":
            self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, peft_config)

        print(" Model hazır!")
        self.model.print_trainable_parameters()
        return self.model, self.tokenizer

    def create_sft_config(self, num_epochs: int = 3, batch_size: int = 4):
        print("\n  SFT konfigürasyonu hazırlanıyor...")

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
            mlm=False
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

        return trainer

    def test_model(self, test_prompts: list):
        """Eğitilmiş modeli test et"""
        print("\n" + "=" * 60)
        print(" MODEL TESTİ")
        print("=" * 60)

        self.model.eval()

        model_device = next(self.model.parameters()).device
        print(f"📍 Model cihazı: {model_device}")

        for prompt in test_prompts:
            print(f"\n📝 Prompt: {prompt}")
            print("-" * 60)

            formatted_prompt = f"### Human: {prompt}\n### Assistant:"

            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(model_device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if "### Assistant:" in response:
                response = response.split("### Assistant:")[-1].strip()

            print(f" Response:\n{response}")
            print("-" * 60)

    def compare_before_after(self, test_prompt: str):
        print("\n" + "=" * 60)
        print(" EĞITIM ÖNCESİ vs SONRASI KARŞILAŞTIRMA")
        print("=" * 60)

        print("\n Orijinal model yükleniyor...")
        original_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )

        original_device = next(original_model.parameters()).device
        print(f"   Orijinal model yüklendi: {original_device}")

        model_device = next(self.model.parameters()).device
        original_device = next(original_model.parameters()).device

        formatted_prompt = f"### Human: {test_prompt}\n### Assistant:"

        print(f"\n Test Prompt: {test_prompt}\n")
        print(" EĞITIM ÖNCESİ (Original Model):")
        print("-" * 60)

        inputs_original = self.tokenizer(formatted_prompt, return_tensors="pt").to(original_device)
        with torch.no_grad():
            outputs_before = original_model.generate(
                **inputs_original, max_new_tokens=150, temperature=0.7, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_before = self.tokenizer.decode(outputs_before[0], skip_special_tokens=True)
        if "### Assistant:" in response_before:
            response_before = response_before.split("### Assistant:")[-1].strip()
        print(response_before)

        print("\n EĞITIM SONRASI (SFT Fine-tuned Model):")
        print("-" * 60)

        inputs_finetuned = self.tokenizer(formatted_prompt, return_tensors="pt").to(model_device)
        with torch.no_grad():
            outputs_after = self.model.generate(
                **inputs_finetuned, max_new_tokens=150, temperature=0.7, do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response_after = self.tokenizer.decode(outputs_after[0], skip_special_tokens=True)
        if "### Assistant:" in response_after:
            response_after = response_after.split("### Assistant:")[-1].strip()
        print(response_after)

        print("\n" + "=" * 60)
        del original_model
        if self.device == "cuda":
            torch.cuda.empty_cache()


def main():
    print("=" * 60)
    print(" SFT (Supervised Fine-Tuning) [Plan C: Standart Trainer]")
    print("=" * 60)

    fine_tuner = SFTFineTuner(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        output_dir="./sft_tinyllama_model"
    )

    model, tokenizer = fine_tuner.setup_model_and_tokenizer(use_quantization=(fine_tuner.device == 'cuda'))

    print("\n Hangi veri setini kullanmak istersiniz?")
    print("  1. Guanaco (OpenAssistant) - Genel talimat takip")
    print("  2. Özel veri seti - Teknik sorular")
    use_custom = True

    if use_custom:
        dataset = fine_tuner.create_custom_dataset()
    else:
        dataset = fine_tuner.load_instruction_dataset(num_samples=500)

    training_args = fine_tuner.create_sft_config(num_epochs=3, batch_size=4)

    trainer = fine_tuner.train(dataset, training_args)

    test_prompts = [
        "Python'da decorator nedir?",
        "REST API tasarlarken nelere dikkat etmeliyim?",
        "Git merge ve rebase arasındaki fark nedir?"
    ]
    fine_tuner.test_model(test_prompts)

    fine_tuner.compare_before_after("Makine öğrenmesinde regularization nedir?")

    print("\n" + "=" * 60)
    print(" SFT Fine-Tuning Tamamlandı!")
    print("=" * 60)
    print(f"\n Model kaydedildi: {fine_tuner.output_dir}")


if __name__ == "__main__":
    main()