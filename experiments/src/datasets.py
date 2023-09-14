import os

DATASET_KEYS = [
    "proverb", "conversation", "explanation",
    "answer1", "answer2", "answer3", "is_figurative", "label"
]


def load_dataset(path, lang="en", is_test=True, is_trans=False):
    fn = "test_proverbs.tsv" if is_test else "fs_proverbs.tsv"
    if is_trans:
        fn = "machine_translation.tsv"
    dataset = {i: [] for i in DATASET_KEYS}
    with open(os.path.join(path, lang, fn), "r") as f:
        for c, l in enumerate(f.readlines()):
            if c == 0 and l.lower().startswith("proverb"):
                continue
            if len(l.strip().split("\t")) != len(DATASET_KEYS):
                print(c, l)
            for i, data in enumerate(l.strip().split("\t")):
                dataset[DATASET_KEYS[i]].append(data.lower())
    for i in DATASET_KEYS:
        print (f"Dataset sanity check: Key {i} - {len(dataset[i])}")
    return dataset


def load_memorized(path):
    group2id = {1: [], 2: []}
    if not path: return group2id
    with open(os.path.join(path), "r") as f:
        for _, l in enumerate(f.readlines()):
            for c, label in enumerate(l.strip().split()):
                group2id[int(label) + 1].append(c)
    return group2id


def get_group2id(labels):
    group2id = {1: [], 2: []}
    for c, l in enumerate(labels):
        group2id[int(l) + 1].append(c)
    return group2id
