import os

import hydra
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import default_data_collator, get_linear_schedule_with_warmup

from src.t5 import MT5ForSequenceClassification

os.environ["TOKENIZERS_PARALLELISM"] = "false"

l2id = {"a": 0, "b": 1}


def prep_dataest(file_path):
    dataset = load_dataset("csv", delimiter="\t", data_files=file_path)
    if "train" in file_path:
        dataset = dataset["train"].train_test_split(shuffle=False, seed=42,
                                                    test_size=20)
        dataset["validation"] = dataset["test"]
    else:
        dataset["test"] = dataset["train"]

    dataset = dataset.map(
        lambda x: {"labels": [l2id.get(l.lower()) for l in x["answer_key"]]},
        batched=True,
        num_proc=1,
    )
    print(dataset["train"][0])
    print(dataset)
    return dataset


def training_loop(model, cfg, tokenizer, dataset, train_dataloader, eval_dataloader):
    model = model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * cfg.num_epochs),
    )
    old_acc = 0
    print(len(train_dataloader))
    for epoch in range(cfg.num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(torch.argmax(outputs.logits, -1).detach().cpu().numpy())

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss).item()
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss).item()
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        accuracy = _eval(eval_preds, dataset["validation"]["answer_key"])
        # if eval_epoch_loss <= old_loss:
        if accuracy >= old_acc:
            old_acc = accuracy
            model_id = f"{cfg.output_path}/{cfg.model_name_or_path}_{cfg.lang}"
            model.save_pretrained(model_id)
            print ("######\nSaving\n#######")


@torch.no_grad()
def eval_loop(model, tokenizer, eval_dataloader):
    model.eval()
    eval_preds = []
    for step, batch in enumerate(eval_dataloader):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        eval_preds.extend(torch.argmax(outputs.logits, -1).detach().cpu().numpy())
    return eval_preds


def _eval(eval_preds, dataset):
    correct = 0
    total = 0
    for pred, label in zip(eval_preds, dataset):
        if pred == l2id[label]:
            correct += 1
        total += 1
    accuracy = correct / total * 100
    print(f"{accuracy=} % on the evaluation dataset")
    print(f"{eval_preds[:10]=}")
    return accuracy


def get_right_patterns(data, cfg, split="train"):
    patterns = ["Context: {conversation} Choices: {answers}"]
    new_inputs = []
    new_labels = []
    for c in range(len(data["answer1"])):
        answers1 = data["answer1"][c]
        answers2 = data["answer2"][c]
        label = data["labels"][c]
        answers = f"A: {answers1} B: {answers2}"

        for pattern in patterns:
            if cfg.pattern_set == "qa_exp":
                new_inputs.append(pattern.format(proverb=data["proverb"][c], conversation=data["conversation"][c],
                                                 explanation=data["explanation"][c], answers=answers).lower())
            else:
                new_inputs.append(pattern.format(proverb=data["proverb"][c], conversation=data["conversation"][c],
                                                 answers=answers).lower())
            new_labels.append(label)

    return new_inputs, new_labels


def _remap_keys(pdir):
    clean_state_dict = dict()
    for k, v in torch.load(pdir).items():
        if "classifier" in k:
            # base_model.model.classifier.original_module.out_proj.weight
            new_k = ".".join(k.split(".")[:-2]) + ".modules_to_save.default." + ".".join(k.split(".")[-2:])
            clean_state_dict[new_k] = v
    return clean_state_dict


@hydra.main(config_path="../configs", config_name="fs_config_std")
def main(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    def preprocess_function_train(examples):
        inputs, targets = get_right_patterns(examples, cfg)
        model_inputs = tokenizer(inputs, max_length=cfg.max_length, padding="max_length", truncation=True,
                                 return_tensors="pt")
        model_inputs["labels"] = targets
        return model_inputs

    def preprocess_function(examples):
        inputs, targets = get_right_patterns(examples, cfg, split="test")
        model_inputs = tokenizer(inputs, max_length=cfg.max_length, padding="max_length", truncation=True,
                                 return_tensors="pt")
        model_inputs["labels"] = targets
        return model_inputs

    dataset = prep_dataest(os.path.join(cfg.data_path, cfg.lang, "transfer_train_proverbs.tsv"))
    if cfg.eval_lang != cfg.lang:
        eval_dataset = prep_dataest(os.path.join(cfg.data_path, cfg.eval_lang, "transfer_test_proverbs.tsv"))
        dataset["validation"] = eval_dataset["validation"]

    train_dataset = dataset["train"].map(
        preprocess_function_train,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    eval_dataset = dataset["validation"].map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=cfg.batch_size, pin_memory=True
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=cfg.batch_size,
                                 pin_memory=True)
    clf_fun = MT5ForSequenceClassification if "t0" in cfg.model_name_or_path else AutoModelForSequenceClassification

    model = clf_fun.from_pretrained(cfg.model_name_or_path, num_labels=2, torch_dtype=torch.bfloat16)

    print(cfg.load_pretrain)
    if cfg.load_pretrain:
        model.load_state_dict(_remap_keys(f"{cfg.load_pretrain}/pytorch_model.bin"), strict=False)

    if cfg.train:
        training_loop(model, cfg, tokenizer, dataset, train_dataloader, eval_dataloader)
    if cfg.eval:
        if cfg.train:
            model.to("cpu")
            del model
            pdir = f"{cfg.output_path}/{cfg.model_name_or_path}_{cfg.lang}"
            model = clf_fun.from_pretrained(pdir, num_labels=2, torch_dtype=torch.bfloat16)

            pdir = f"{cfg.output_path}/{cfg.model_name_or_path}_{cfg.lang}/pytorch_model.bin"
            model.load_state_dict(torch.load(pdir), strict=False)

        model.to("cuda")
        model.eval()

        print("\n\n########\nFinal Eval on Train\n########")
        eval_preds = eval_loop(model, tokenizer, train_dataloader)
        _eval(eval_preds, dataset["train"]["answer_key"])
        print("\n\n########\nFinal Eval\n########")
        eval_preds = eval_loop(model, tokenizer, eval_dataloader)
        _eval(eval_preds, dataset["validation"]["answer_key"])

    if cfg.test:
        if cfg.train:
            model.to("cpu")
            del model
            pdir = f"{cfg.output_path}/{cfg.model_name_or_path}_{cfg.lang}"
            model = clf_fun.from_pretrained(pdir, num_labels=2, torch_dtype=torch.float16)

        model.to("cuda")
        model.eval()
        results = {}
        for lang in cfg.test_lang:
            fn = "transfer_test_proverbs.tsv" if lang == "en" else "test_proverbs.tsv"
            print(fn, lang)
            test_dataset = prep_dataest(os.path.join(cfg.data_path, lang, fn))["test"]
            test_labels = test_dataset["answer_key"]
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=1,
                remove_columns=dataset["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataloader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=cfg.batch_size,
                                         pin_memory=True)
            print(f"\n\n########\nTest {lang}\n########")
            eval_preds = eval_loop(model, tokenizer, eval_dataloader)
            accuracy = _eval(eval_preds, test_labels)
            results[lang] = accuracy
        output_path = f"{cfg.output_path}/{cfg.model_name_or_path}_results_s{cfg.seed}.txt"
        with open(output_path, 'a') as f:
            for k, v in results.items():
                f.write(f"{k}\t{v}\n")
        print(results)


if __name__ == '__main__':
    main()
