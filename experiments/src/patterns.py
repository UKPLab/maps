import numpy as np

MEM_PATTERNS = {
    "en": [
        "proverb: {proverb}",
        "complete this proverb: {proverb}",
        "finish the proverb: {proverb}",
        "what's missing at the end of this proverb: {proverb}",
        "what's the last word of this proverb: {proverb}",
    ],

    "de": [
        "Sprichwort: {proverb}",
        "Vervollständige dieses Sprichwort: {proverb}",
        "Beende das Sprichwort: {proverb}",
        "Was fehlt am Ende dieses Sprichworts: {proverb}",
        "Was ist das letzte Wort dieses Sprichworts: {proverb}"
    ],

    "bn": [
        "প্রবাদ: {proverb}",
        "এই প্রবাদটি সম্পূর্ণ করো: {proverb}",
        "প্রবাদটি শেষ করো: {proverb}",
        "এই প্রবাদের শেষে কি অনুপস্থিত: {proverb}",
        "এই প্রবাদের শেষ শব্দ কি: {proverb}"
    ],

    "ru": [
        "пословицу: {proverb}",
        "Завершите эту пословицу: {proverb}",
        "Закончите пословицу: {proverb}",
        "Что отсутствует в конце этой пословицы: {proverb}",
        "Какое последнее слово в этой пословице: {proverb}"
    ],

    "zh": [
        "谚语: {proverb}",
        "补全这个谚语：{proverb}",
        "完成这个谚语：{proverb}",
        "这个谚语结尾缺什么字：{proverb}",
        "这个谚语的最后一个字是什么：{proverb}"
    ],

    "id": [
        "peribahasa: {proverb}",
        "lengkapi peribahasa ini: {proverb}",
        "selesaikan peribahasa ini: {proverb}",
        "apa yang hilang di akhir peribahasa ini: {proverb}",
        "apa kata terakhir peribahasa ini: {proverb}"
    ]
}

QA_PATTERNS_NEG = {
    "en": [
        "Question: What does the person not mean by the proverb?\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}
QA_PATTERNS_POS = {
    "en": [
        "Question: What does the person mean by the proverb?\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}

QA_PATTERNS_NEG1 = {
    "en": [
        "Question: Which answer is contrary to what the person means by the proverb?\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}

QA_PATTERNS_NEG2 = {
    "en": [
        "Question: Which answer is impossible as the interpretation of what the person means by the proverb?\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}

QA_PATTERNS_NEG3 = {
    "en": [
        "Question: Pick the opposite answer to what the person means by the proverb.\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}
QA_PATTERNS_NEG4 = {
    "en": [
        "Question: Pick the wrong answer to what the person means by the proverb.\nProverb:'{proverb}'\nContext: {conversation}\nChoices: {answers}\nAnswer:",
    ]
}

QA_PATTERNS_MLM_FS = {
    "en": [
        "Proverb:'{proverb}' Context: {conversation} Choices: {answers}",
    ],
}

QA_PATTERNS_MLM_EXP_FS = {
    "en": [
        "Proverb:'{proverb}' Meaning:{explanation} Context: {conversation} Choices: {answers}",
    ],
}

PATTERNS_DICT = {
    "qa_ppl_neg": QA_PATTERNS_NEG,
    "qa_ppl_pos": QA_PATTERNS_POS,
    "qa_ppl_neg1": QA_PATTERNS_NEG1,
    "qa_ppl_neg2": QA_PATTERNS_NEG2,
    "qa_ppl_neg3": QA_PATTERNS_NEG3,
    "qa_ppl_neg4": QA_PATTERNS_NEG4,
}


def get_k_shot_patterns(cfg, pat_lang):
    from .datasets import load_dataset
    np.random.seed(cfg.seed)
    if "pos" in cfg.pattern_set:
        prompt = QA_PATTERNS_POS["en"][0]
    else:
        prompt = QA_PATTERNS_NEG["en"][0]
    patterns = []
    datasets = load_dataset(cfg.data_path, lang=pat_lang, is_test=False)
    proverbs = datasets["proverb"]
    conversation = datasets["conversation"]
    answers1 = datasets["answer1"]
    answers2 = datasets["answer2"]
    labels = datasets["label"]

    for n in range(5):
        n_shot = []
        indices = np.random.permutation(30)

        for c in indices[:cfg.n_shot]:
            answers = f"A: {answers1[c]} B: {answers2[c]}"
            inputs = prompt.format(proverb=proverbs[c], conversation=conversation[c], answers=answers).lower()
            if "neg" in cfg.pattern_set:
                label = "b" if labels[c] == "a" else "a"
            else:
                label = labels[c]
            n_shot.append(inputs + f" {label}\n")
        n_shot.append(prompt)

        patterns.append("\n".join(n_shot))

    print(patterns)
    np.random.seed(0)
    return patterns
