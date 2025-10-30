import json
from datasets import load_dataset, Dataset


def load_instruction_dataset(dataset_name: str, num_samples: int = 1000):
    """
    Veri setini Hugging Face Hub'dan yükle
    """
    print(f"\nVeri seti yükleniyor: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f" {len(dataset['train'])} eğitim, {len(dataset['test'])} test örneği yüklendi")
    if len(dataset['train']) > 0:
        print(f"\nÖrnek veri (birleştirilmiş 'text' formatı):")
        print(dataset['train'][0]['text'][:300] + "...")
    return dataset


def create_custom_dataset(data_path: str, tokenizer):
    """
    Özel talimat veri setini .jsonl dosyasından oluştur
    """
    print(f"\nÖzel veri seti oluşturuluyor: {data_path}...")

    if tokenizer is None:
        raise ValueError("HATA: Tokenizer None olamaz. Önce setup_model_and_tokenizer çağrılmalı.")

    formatted_data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            # Chat formatı
            text = f"### Human: {item['instruction']}\n### Assistant: {item['response']}"
            text += tokenizer.eos_token  # Cümle sonu token'ı ekle
            formatted_data.append({"text": text})

    if not formatted_data:
        raise ValueError(f"{data_path} dosyasında veri bulunamadı veya okunamadı.")

    dataset = Dataset.from_list(formatted_data)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    print(f" {len(dataset['train'])} eğitim, {len(dataset['test'])} test örneği oluşturuldu")
    return dataset