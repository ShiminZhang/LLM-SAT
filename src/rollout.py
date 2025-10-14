from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import json

def main():
    base_model = "./models/gpt-oss-20b"
    adapter = "./gpt_oss_dpo_output_bs8/checkpoint-125"

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.to(device)
    model.eval()

    with open("dataset/rollout.json", "r") as f:
        data = json.load(f)
        encoded = tokenizer(
            data["prompt"],
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
        ).to(device)
        outputs = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_length=6000,
        )
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
