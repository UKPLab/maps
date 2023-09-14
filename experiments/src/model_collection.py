from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForMaskedLM

MODEL_LIST_S = [
    "bigscience/bloomz-560m",
    "bigscience/mt0-base",
    "google/mt5-base"
]

MODEL_LIST_M = [
    "bigscience/bloomz-3b",
    "bigscience/mt0-xl",
    "google/mt5-xl"
]

MODEL_LIST_L = [
    "bigscience/bloomz-7b1",
    "llama-2-hf/7B",
]

MODEL_LIST_XL = [
    "llama-2-hf/13B",
    "bigscience/mt0-xxl",
    "google/mt5-xxl",
]

MODEL_LIST_MT = [
    "bigscience/bloomz-7b1-mt",
    "bigscience/mt0-xxl-mt",
]

MODEL_LIST_MLM = [
    "xlm-roberta-base",
    "xlm-roberta-large",
    "facebook/xlm-roberta-xl",
]

MODEL_LIST_BLOOM = [
    "bigscience/bloomz-560m",
    "bigscience/bloomz-3b",
    "bigscience/bloomz-7b1",
]

MODEL_LIST_MT0 = [
    "bigscience/mt0-base",
    "bigscience/mt0-xl",
    "bigscience/mt0-xxl",
]

MODEL_LIST_XGLM = [
    "facebook/xglm-564M",
    "facebook/xglm-2.9B",
    "facebook/xglm-7.5B",
]

MODEL_LIST_LLAMA = [
    "llama-2-hf/7B",
    "llama-2-hf/13B",
]

SIZE2MODEL = {
    "s": MODEL_LIST_S,
    "m": MODEL_LIST_M,
    "l": MODEL_LIST_L,
    "xl": MODEL_LIST_XL,
    "mt": MODEL_LIST_MT,
    "llama": MODEL_LIST_LLAMA,
    "mt0": MODEL_LIST_MT0,
    "mlm": MODEL_LIST_MLM,
    "xglm": MODEL_LIST_XGLM,
    "bloom": MODEL_LIST_BLOOM,
}


def masking_last_word(proverb, lang, model_name=None):
    if lang != "zh":
        words = proverb.split(" ")
        front = " ".join(words[:-1])
        back = words[-1]
        if "xlm" in model_name:
            front += " <mask>"
    else:
        front = proverb[:-1]
        back = proverb[-1]
        if "xlm" in model_name:
            front += "<mask>"
    return front, back


def perturb_last_word(proverb, lang, model_name=None):
    if lang != "zh":
        words = proverb.split(" ")
        front = " ".join(words[:-1])
        front += + " " + words[-2]

    else:
        front = proverb[:-1]
        back = proverb[-2]
        front += back

    return [front]


def load_model(model_name="mt5"):
    if "llama-2" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     device_map="auto",
                                                     load_in_8bit=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "mt5" in model_name or "mt0" in model_name:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
    elif "xlm" in model_name:
        model = AutoModelForMaskedLM.from_pretrained(model_name).to("cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True,
                                                     trust_remote_code=True)

    return tokenizer, model


def prob_mlm(inputs, tokenizer, model):
    logits = model(**inputs).logits
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
    return tokenizer.decode(predicted_token_id)


def prob_mlm_ppl(inputs, tokenizer, model, label):
    logits = model(**inputs).logits
    label_id = tokenizer.convert_tokens_to_ids(label)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    logit = logits[0, mask_token_index][0, label_id]
    return logit
