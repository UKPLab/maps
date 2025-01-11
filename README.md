# Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings 
<p  align="center">
  <img src='logo.png' width='200'>
</p>

This repository contains our code for experiments and experiment logs.

To access the data, please fill the form [here](https://forms.gle/rJLYcSvD5MooRowe6).
A link to our dataset will show automatically after submit the form.
 
If you encounter any issues or have suggestions, please do not hesitate to email us at:
chen.liu AT tu-darmstadt DOT de


https://www.tu-darmstadt.de/

https://www.ukp.tu-darmstadt.de/

>This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.

### Setup
Install all Python requirements listed in `requirements.txt` (check [here](https://pytorch.org/get-started/locally/) 
to see how to install Pytorch on your system).

You can install the requirements.txt like:
```
pip install --upgrade pip
pip install -r requirements.txt
```

### Preparations
Please put the data into the dataset folder.

### Usage

- Memorization experiment: `memorization.py`.
- Inference experiment: `qa_experiments.py`.
- Fine-tuning baseline: `ft_standard.py`.

To run:
```
python EXPERIMENT_NAME.py
```

The corresponding hyperparameters are in the `configs` folder. 
Each experiment entry point has its own config file, managed through Hydra.

To add models, checkout the `model_collection.py`.
To change prompt patterns, checkout the `patterns.py`.

## Citation
If you find this repository helpful, consider to cite the following paper:

```
@inproceedings{liu-etal-2024-multilingual,
    title = "Are Multilingual {LLM}s Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings",
    author = "Liu, Chen  and
      Koto, Fajri  and
      Baldwin, Timothy  and
      Gurevych, Iryna",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.112/",
    doi = "10.18653/v1/2024.naacl-long.112",
    pages = "2016--2039"
}
```
