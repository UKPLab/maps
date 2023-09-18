# Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings 

This repository contains our code for experiments and experiment logs.

Our paper is currently under review, we will release the dataset at a later time.

🗺️ If you would like to access to the dataset, please don't hesitate to email us at:
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
If you find this repository helpful, feel free to cite the following paper:

```
@misc{liu2023multilingual,
      title={Are Multilingual LLMs Culturally-Diverse Reasoners? An Investigation into Multicultural Proverbs and Sayings}, 
      author={Chen Cecilia Liu and Fajri Koto and Timothy Baldwin and Iryna Gurevych},
      year={2023},
      eprint={2309.08591},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```