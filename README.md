# SFT (Supervised Fine-Tuning) ile TinyLlama Eğitimi

Bu proje, Hugging Face `transformers` ve `peft` (QLoRA) kütüphanelerini kullanarak `TinyLlama-1.1B-Chat-v1.0` modelini (veya benzeri bir modeli) talimat verileriyle (instruction data) ince ayar (fine-tuning) yapmak için kullanılır.

Kod, `transformers.Trainer` sınıfını temel alır ve `trl` kütüphanesinin `SFTTrainer` sınıfına bir alternatif olarak çalışır.

##  Özellikler

* **QLoRA:** 4-bit kuantizasyon ile (CUDA GPU varsa) verimli eğitim.
* **Modüler Yapı:** Veri işleme, model sınıfı ve eğitim script'i birbirinden ayrılmıştır.
* **Esnek Veri Yükleme:** Hugging Face Hub'dan (`timdettmers/openassistant-guanaco`) veya yerel `.jsonl` dosyasından özel veri yükleme.
* **CLI Desteği:** `argparse` ile model, veri seti, epoch gibi parametreleri komut satırından kolayca değiştirme.
* **Test Script'leri:** Eğitim sonrası modeli test etmek için `train.py` içinde ve ayrı bir `inference.py` script'i.


## 🛠️ Kurulum

1.  **Depoyu Klonlama:**
    ```bash
    git clone [https://github.com/AbdulSametTurkmenoglu/sft_fine_tuning.git](https://github.com/AbdulSametTurkmenoglu/sft_fine_tuning.git)
    cd sft_fine_tuning
    ```

2.  **Sanal Ortam :**
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Yükleme:**
    ```bash
    pip install -r requirements.txt
    ```
    *Not: CUDA ortamı için `torch` kurulumunuzun GPU destekli olduğundan emin olun.*

##  Kullanım

### 1. Modeli Eğitme

Ana eğitim script'i `train.py`'dir.

**Örnek 1: Özel `custom_instructions.jsonl` verisi ile eğitim (Varsayılan):**
(Bu, `data/` klasöründeki teknik Q&A verisini kullanır)

```bash
python train.py
```

**Örnek 2: Hugging Face Hub'dan "Guanaco" verisi ile eğitim:**

```bash
python train.py --dataset guanaco --guanaco_samples 1000 --output_dir sft_guanaco_model
```

**Örnek 3: Farklı bir model ve epoch sayısı ile eğitim:**

```bash
python train.py --model_name "google/gemma-2b" --epochs 5 --output_dir sft_gemma_model
```

Tüm argümanları görmek için:
```bash
python train.py --help
```

### 2. Eğitilmiş Model ile Çıkarım (Inference)

Eğitim tamamlandığında (`sft_tinyllama_model` klasörü oluştuğunda), `inference.py` script'ini kullanarak modelle sohbet edebilirsiniz.

Bu script, LoRA adaptörünü ana modelle birleştirir (`merge_and_unload`) ve hızlı çıkarım sağlar.

```bash
python inference.py --model_path sft_tinyllama_model
```

Eğer farklı bir model veya çıktı klasörü kullandıysanız:
```bash
python inference.py --base_model "google/gemma-2b" --model_path "sft_gemma_model"
```