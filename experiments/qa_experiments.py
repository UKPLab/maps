import os

import hydra
import torch

from src.datasets import load_dataset, load_memorized, get_group2id
from src.model_collection import *
from src.patterns import PATTERNS_DICT, get_k_shot_patterns
from utils import generate_statistics_and_report, generate_all, generate_group_statistics_and_report

torch.manual_seed(0)
import numpy as np

np.random.seed(0)
import re


@torch.no_grad()
def qa_ppl(datasets, patterns, tokenizer, model, pattern_set,
           model_name="mt5", lang="en",
           output_path="./experiment_logs", groupings=dict()):
    proverbs = datasets["proverb"]
    conversation = datasets["conversation"]
    explanations = datasets["explanation"]
    answers1 = datasets["answer1"]
    answers2 = datasets["answer2"]
    labels = datasets["label"]
    print(f"{len(proverbs)} proverbs to inference on")

    total = len(proverbs)
    all_acc = []
    id2group, grouping_acc = prep_groups(groupings)

    for pattern in patterns:
        correct, incorrect = 0, 0
        for c, proverb in enumerate(proverbs):
            answers = f"A: {answers1[c]} B: {answers2[c]}"
            if "explanation" in pattern:
                inputs = pattern.format(proverb=proverb, conversation=conversation[c], explanation=explanations[c],
                                        answers=answers).lower()
            else:
                inputs = pattern.format(proverb=proverb, conversation=conversation[c], answers=answers).lower()
            inputs = re.sub('[ \t]{2,}', ' ', inputs).replace('\n ', '\n')
            output_a, output_b = _probing_logic(tokenizer, model, inputs, model_name)

            if output_a < output_b:
                generated = "b"
            else:
                generated = "a"

            groupid = id2group.get(c, 1)
            if generated == labels[c]:
                if "neg" in pattern_set:
                    incorrect += 1
                    grouping_acc[groupid][1] += 1
                else:
                    correct += 1
                    grouping_acc[groupid][0] += 1

            else:
                if "neg" in pattern_set:
                    correct += 1
                    grouping_acc[groupid][0] += 1
                else:
                    # print(inputs)
                    incorrect += 1
                    grouping_acc[groupid][1] += 1

            if c % 50 == 0:
                print(pattern_set)
                print(inputs)
                print(f"\nGen: {generated} L:{labels[c]}")
                print(f"Current item: {c}")
        acc = generate_statistics_and_report(total, correct, incorrect,
                                             pattern, model_name, output_path)
        generate_group_statistics_and_report(grouping_acc, model_name, output_path)

        all_acc.append(acc)
    generate_all(all_acc, model_name, lang, output_path)


def get_logits(model, tokenizer, prompt, label_ids=None, label_attn=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    outputs = model(**inputs, decoder_input_ids=model._shift_right(label_ids))
    logits = outputs.logits
    logprobs = torch.gather(logits, 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
    return logprobs.sum() / label_attn.sum()


def seq2seq(model, tokenizer, input_text, labels):
    labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt').to("cuda")
    list_label_ids = labels_encoded['input_ids'].to('cuda')
    list_label_attn = labels_encoded['attention_mask'].to('cuda')
    probs = [
        get_logits(model, tokenizer, input_text, label_ids.view(1, -1), label_attn.view(1, -1))
        for (label_ids, label_attn) in zip(list_label_ids, list_label_attn)
    ]
    return probs[0], probs[1]


def _probing_logic(tokenizer, model, inputs, model_name):
    if "xlm" in model_name:
        inputs_id = tokenizer(inputs + " <mask>", return_tensors="pt").to("cuda")
        logits = model(**inputs_id).logits
        label_a = tokenizer.convert_tokens_to_ids("a")
        label_b = tokenizer.convert_tokens_to_ids("b")
        mask_token_index = (inputs_id.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        logit_a = logits[0, mask_token_index][0, label_a]
        logit_b = logits[0, mask_token_index][0, label_b]
        probs = torch.softmax(torch.tensor([logit_a, logit_b]), dim=0)
        output_a = probs[0]
        output_b = probs[1]
    elif "xglm" in model_name:
        a = re.sub('\n+', '\n', f'{inputs}\na').replace("\n", " <\s>")
        b = re.sub('\n+', '\n', f'{inputs}\nb').replace("\n", " <\s>")
        labels_a = tokenizer(a, return_token_type_ids=False,
                             return_tensors="pt", max_length=1024).to("cuda")
        labels_b = tokenizer(b, return_token_type_ids=False,
                             return_tensors="pt", max_length=1024).to("cuda")
        output_a = model(input_ids=labels_a["input_ids"], labels=labels_a["input_ids"]).logits
        output_b = model(input_ids=labels_b["input_ids"], labels=labels_b["input_ids"]).logits
        output_a = torch.gather(output_a, 2, labels_a["input_ids"][:, 1:].unsqueeze(2)).mean().item()
        output_b = torch.gather(output_b, 2, labels_b["input_ids"][:, 1:].unsqueeze(2)).mean().item()
    elif "mt0" in model_name:
        output_a, output_b = seq2seq(model, tokenizer, inputs, ["a", "b"])
    elif "bloomz" in model_name or "llama" in model_name:
        labels_a = tokenizer(f'{inputs} a', return_token_type_ids=False,
                             return_tensors="pt", max_length=1024).to("cuda")
        labels_b = tokenizer(f'{inputs} b', return_token_type_ids=False,
                             return_tensors="pt", max_length=1024).to("cuda")

        output_a = model(input_ids=labels_a["input_ids"], labels=labels_a["input_ids"]).logits
        output_b = model(input_ids=labels_b["input_ids"], labels=labels_b["input_ids"]).logits
        output_a = torch.gather(output_a, 2, labels_a["input_ids"][:, 1:].unsqueeze(2)).mean().item()
        output_b = torch.gather(output_b, 2, labels_b["input_ids"][:, 1:].unsqueeze(2)).mean().item()

    return output_a, output_b


def prep_groups(groupings):
    id2group = dict()
    grouping_acc = {1: [0, 0], 2: [0, 0]}
    for k, v in groupings.items():
        for i in v:
            id2group[i] = k
    return id2group, grouping_acc


@hydra.main(config_path="../configs", config_name="qa_config")
def main(cfg):
    if not os.path.isdir(f"{cfg.output_path}"):
        os.mkdir(f"{cfg.output_path}")
    if not os.path.isdir(f"{cfg.output_path}/{cfg.pattern_set}"):
        os.mkdir(f"{cfg.output_path}/{cfg.pattern_set}")

    for model_name in SIZE2MODEL.get(cfg.model_size):
        tokenizer, model = load_model(model_name=model_name)
        print(f"Now testing on: {model_name}")
        for lang in cfg.lang:
            datasets = load_dataset(cfg.data_path, lang=lang, is_trans=cfg.trans_test)
            print(f"Loaded {lang} proverbs.")
            pat_lang = "en"
            mn = model_name.split("/")[-1]
            model.eval()
            output_path = f"{cfg.output_path}/{cfg.pattern_set}/{pat_lang}pat_{mn}_{cfg.model_size}_{lang}_qa.log"

            if "nshot" in cfg.pattern_set:
                patterns = get_k_shot_patterns(cfg, pat_lang)
                output_path = f"{cfg.output_path}/{cfg.pattern_set}/{cfg.n_shot}_{mn}_{cfg.model_size}_{lang}_qa.log"

            else:
                patterns = PATTERNS_DICT.get(cfg.pattern_set)[pat_lang]

            if cfg.mem_file:
                groupings = load_memorized(cfg.mem_file)
            else:
                groupings = get_group2id(datasets["is_figurative"])
            qa_ppl(datasets=datasets, patterns=patterns,
                   tokenizer=tokenizer, model=model, pattern_set=cfg.pattern_set,
                   model_name=model_name, lang=lang,
                   output_path=output_path, groupings=groupings)


if __name__ == '__main__':
    main()
