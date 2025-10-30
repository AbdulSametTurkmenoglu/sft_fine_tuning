import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.fine_tuner import SFTFineTuner
from src.data_utils import load_instruction_dataset, create_custom_dataset


def test_model(model, tokenizer, test_prompts: list):
    """Eğitilmiş modeli test et"""
    print("\n" + "=" * 60)
    print(" MODEL TESTİ (Eğitim Sonrası)")
    print("=" * 60)

    model.eval()
    model_device = next(model.parameters()).device
    print(f" Model cihazı: {model_device}")

    for prompt in test_prompts:
        print(f"\n Prompt: {prompt}")
        print("-" * 60)

        formatted_prompt = f"### Human: {prompt}\n### Assistant:"
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt"
        ).to(model_device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "### Assistant:" in response:
            response = response.split("### Assistant:")[-1].strip()

        print(f" Response:\n{response}")
        print("-" * 60)


def compare_before_after(finetuned_model, tokenizer, model_name: str, test_prompt: str):
    """Eğitim öncesi ve sonrası modeli karşılaştır"""
    print("\n" + "=" * 60)
    print(" EĞITIM ÖNCESİ vs SONRASI KARŞILAŞTIRMA")
    print("=" * 60)

    print("\n Orijinal (base) model yükleniyor...")

    device_type = next(finetuned_model.parameters()).device.type
    dtype = torch.float16 if device_type == "cuda" else torch.float32

    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=dtype,
    )
    original_device = next(original_model.parameters()).device
    print(f"   Orijinal model yüklendi: {original_device}")

    finetuned_device = next(finetuned_model.parameters()).device
    formatted_prompt = f"### Human: {test_prompt}\n### Assistant:"

    print(f"\n Test Prompt: {test_prompt}\n")
    print(" EĞITIM ÖNCESİ (Original Model):")
    print("-" * 60)

    inputs_original = tokenizer(formatted_prompt, return_tensors="pt").to(original_device)
    with torch.no_grad():
        outputs_before = original_model.generate(
            **inputs_original, max_new_tokens=150, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response_before = tokenizer.decode(outputs_before[0], skip_special_tokens=True)
    if "### Assistant:" in response_before:
        response_before = response_before.split("### Assistant:")[-1].strip()
    print(response_before)

    print("\n EĞITIM SONRASI (SFT Fine-tuned Model):")
    print("-" * 60)
    finetuned_model.eval()
    inputs_finetuned = tokenizer(formatted_prompt, return_tensors="pt").to(finetuned_device)
    with torch.no_grad():
        outputs_after = finetuned_model.generate(
            **inputs_finetuned, max_new_tokens=150, temperature=0.7, do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response_after = tokenizer.decode(outputs_after[0], skip_special_tokens=True)
    if "### Assistant:" in response_after:
        response_after = response_after.split("### Assistant:")[-1].strip()
    print(response_after)

    print("\n" + "=" * 60)
    del original_model
    if device_type == "cuda":
        torch.cuda.empty_cache()



def main():
    parser = argparse.ArgumentParser(description="SFT Fine-Tuning Script")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Hugging Face model adı")
    parser.add_argument("--output_dir", type=str, default="sft_tinyllama_model",
                        help="Eğitilmiş modelin kaydedileceği dizin")
    parser.add_argument("--dataset", type=str, default="custom", choices=['custom', 'guanaco'],
                        help="Kullanılacak veri seti: 'custom' (yerel) veya 'guanaco' (hub)")
    parser.add_argument("--custom_data_path", type=str, default="data/custom_instructions.jsonl",
                        help="Özel veri setinin yolu (.jsonl)")
    parser.add_argument("--guanaco_samples", type=int, default=500, help="Guanaco veri setinden alınacak örnek sayısı")
    parser.add_argument("--epochs", type=int, default=3, help="Eğitim epoch sayısı")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch boyutu")
    parser.add_argument("--no_quantization", action="store_true", help="4-bit quantization'ı devre dışı bırak")

    args = parser.parse_args()

    print("=" * 60)
    print(" SFT (Supervised Fine-Tuning) Başlatılıyor")
    print("=" * 60)

    fine_tuner = SFTFineTuner(
        model_name=args.model_name,
        output_dir=args.output_dir
    )

    use_quant = (fine_tuner.device == 'cuda') and not args.no_quantization
    model, tokenizer = fine_tuner.setup_model_and_tokenizer(use_quantization=use_quant)

    if args.dataset == "custom":
        dataset = create_custom_dataset(args.custom_data_path, tokenizer)
    else:
        dataset = load_instruction_dataset("timdettmers/openassistant-guanaco", num_samples=args.guanaco_samples)

    training_args = fine_tuner.create_sft_config(num_epochs=args.epochs, batch_size=args.batch_size)

    trainer = fine_tuner.train(dataset, training_args)

    test_prompts = [
        "Python'da decorator nedir?",
        "REST API tasarlarken nelere dikkat etmeliyim?",
        "Git merge ve rebase arasındaki fark nedir?"
    ]
    test_model(trainer.model, tokenizer, test_prompts)

    compare_before_after(trainer.model, tokenizer, args.model_name, "Makine öğrenmesinde regularization nedir?")

    print("\n" + "=" * 60)
    print(" SFT Fine-Tuning Tamamlandı!")
    print("=" * 60)
    print(f"\n Model kaydedildi: {fine_tuner.output_dir}")


if __name__ == "__main__":
    main()