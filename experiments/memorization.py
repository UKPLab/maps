import torch
from src.datasets import load_dataset
from src.patterns import MEM_PATTERNS
from src.model_collection import *
from utils import generate_statistics_and_report, generate_all
import hydra

torch.manual_seed(0)
import numpy as np

np.random.seed(0)

PATTERNS_DICT = {
    "memorization": MEM_PATTERNS,
}


@torch.no_grad()
def memorization(datasets, patterns, tokenizer, model,
                 model_name="mt5", lang="en", output_path="./experiment_logs"):
    proverbs = datasets["proverb"]
    print (f"{len(proverbs)} proverbs to inference on")

    total = len(proverbs)
    all_acc = []
    is_memorized = [0] * len(proverbs)
    for pattern in patterns:
        correct, incorrect = 0, 0
        for c, proverb in enumerate(proverbs):
            front, last_word = masking_last_word(proverb, lang, model_name)
            inputs = pattern.format(proverb=front).lower()
            if "xlm" in model_name:
                inputs = tokenizer(inputs, return_tensors="pt").to("cuda")
                generated = prob_mlm(inputs, tokenizer, model)
            elif "mt5" in model_name or "mt0" in model_name:
                inputs = tokenizer(inputs, return_token_type_ids=False, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs)
                generated = tokenizer.decode(outputs[0]).replace("</s>", "")
            else:
                inputs = tokenizer(inputs, return_token_type_ids=False, return_tensors="pt").to("cuda")
                outputs = model.generate(**inputs, max_new_tokens=4)
                generated = tokenizer.decode(outputs[0]).replace("</s>", "")

            g = []
            generated = generated.replace(">", "> ")
            generated = generated.replace(pattern.split(":")[0] + ":", "").strip()
            for tok in generated.split(" "):
                if ">" not in tok:
                    g.append(tok.strip())
            generated = " ".join(g).strip().lower().strip(".")
            if "mt5" in model_name or "mt0" in model_name:
                if lang == "zh":
                    generated = front + generated.lower()
                else:
                    generated = front + " " + generated.lower()
            if "xlm" in model_name:
                if lang == "zh":
                    generated = front.replace("<mask>", "").strip() + generated.lower()
                else:
                    generated = front.replace("<mask>", "").strip() + " " + generated.lower()

            input_pat = pattern.format(proverb=proverb).lower()
            if generated != input_pat:
                if proverb in generated:
                    correct += 1
                    is_memorized[c] = 1
                else:
                    if generated == proverb:
                        correct += 1
                        is_memorized[c] = 1
                    else:
                        incorrect += 1
            else:
                correct += 1
                is_memorized[c] = 1

            if c % 50 == 0:
                print(proverb.lower())
                print(generated)
                print(f"Current item: {c}, label {is_memorized[c]}")

        acc = generate_statistics_and_report(total, correct, incorrect,
                                             pattern, model_name, output_path + "_memorization.log")
        all_acc.append(acc)

    with open(output_path + "_labels.log", 'w') as f:
        f.write(" ".join([str(i) for i in is_memorized]))
    f.close()

    generate_all(all_acc, model_name, lang, output_path + "_memorization.log", sum(is_memorized) / len(is_memorized))


@hydra.main(config_path="../configs", config_name="mem_config")
def main(cfg):
    for model_name in SIZE2MODEL.get(cfg.model_size):
        tokenizer, model = load_model(model_name=model_name)
        print(f"Now testing on: {model_name}")
        for lang in cfg.lang:
            datasets = load_dataset(cfg.data_path, lang=lang)
            print(f"Loaded {lang} proverbs.")
            pat_lang = "en" if cfg.model_size not in ["mt", 'mlm'] else lang
            patterns = PATTERNS_DICT.get(cfg.pattern_set)[pat_lang]
            mn = model_name.split("/")[-1]
            output_path = f"{cfg.output_path}/{pat_lang}pat_{mn}_{cfg.model_size}_{lang}"
            model.eval()
            memorization(datasets=datasets, patterns=patterns,
                         tokenizer=tokenizer, model=model,
                         model_name=model_name, lang=lang,
                         output_path=output_path)


if __name__ == '__main__':
    main()
