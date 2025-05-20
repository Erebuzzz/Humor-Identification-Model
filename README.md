# Humor Identification in Yelp Reviews

## Overview

This project implements a machine learning pipeline for automatic humor detection in Yelp reviews. It combines graph-based features (all 9 Zagreb indices), lexical ambiguity, stylistic cues, and multiple word embeddings (GloVe, Word2Vec, BERT) with classical and neural models, including ensemble learning.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Sources](#data-sources)
- [Feature Engineering](#feature-engineering)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Project Structure

```
.
├── HIM.py                    # Main pipeline script
├── glove.6B.100d.txt         # GloVe embeddings (downloaded)
├── bert-base-uncased/        # Local BERT model files (downloaded)
├── yelp_academic_dataset_review.json  # Yelp reviews dataset (downloaded)
├── zagreb_indices_results.txt # Exported Zagreb indices (output)
├── instructions.md           # Local file setup instructions
└── README.md                 # This file
```


---

## Setup and Installation

1. **Clone the repository and create a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
```

2. **Install Python dependencies:**

```bash
pip install numpy scipy scikit-learn tqdm networkx nltk gensim torch transformers
```

3. **Download required files (see [instructions.md](instructions.md)):**
    - Yelp reviews: `yelp_academic_dataset_review.json`
    - GloVe embeddings: `glove.6B.100d.txt`
    - BERT model files: Place all files from [bert-base-uncased](https://huggingface.co/bert-base-uncased) in a folder named `bert-base-uncased/`
4. **Ensure all paths in your code (e.g., `DATASET_PATH`, `GLOVE_PATH`, `BERT_LOCAL_PATH`) are correct.**

---

## Data Sources

- **Yelp Open Dataset:** [https://www.yelp.com/dataset](https://www.yelp.com/dataset)
- **GloVe Embeddings:** [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
- **BERT Model:** [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased)

---

## Feature Engineering

- **Graph-based:** All 9 Zagreb indices from word co-occurrence graphs
- **Lexical ambiguity:** WordNet synset statistics (mean, max, gap)
- **Stylistic:** Text length, exclamation/question marks
- **Word embeddings:** GloVe (pre-trained), Word2Vec (trained in-domain), BERT (contextual)

---

## Modeling and Evaluation

- **Classical Models:** SVM, Naive Bayes
- **Neural Model:** PyTorch MLP (Adam and RMSprop optimizers)
- **Ensemble:** Stacking ensemble (SVM, NB, Random Forest meta-classifier)
- **Metrics:** Accuracy, precision, recall, F1-score

---

## How to Run

1. **Prepare all required files as described above.**
2. **Activate your virtual environment.**
3. **Run the main script:**

```bash
python HIM.py
```

4. **After running, check `zagreb_indices_results.txt` for exported Zagreb indices.**

---

## Results

- **Best accuracy:** 76.67% (SVM)
- **Best F1-score:** 77.03% (SVM)
- **Stacking ensemble:** 75.40% accuracy, 75.73% F1-score
- **All 9 Zagreb indices exported for selected reviews**

See the Results section in your report or thesis for full tables and discussion.

---

## References

- [Yelp Open Dataset](https://www.yelp.com/dataset)
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Hugging Face BERT](https://huggingface.co/bert-base-uncased)
- Mahajan \& Zaveri, 2024. "An automatic humor identification model with novel features from Berger's typology and ensemble models." *Intelligent Systems with Applications*, 2024.

---

**For detailed setup, see [instructions.md](instructions.md).**

---

<div style="text-align: center">⁂</div>

[^1]: paste.txt

[^2]: paste-2.txt

[^3]: https://github.com/catiaspsilva/README-template

[^4]: https://www.techtarget.com/searchsoftwarequality/tip/How-to-create-an-engaging-README-file

[^5]: https://github.com/sfbrigade/data-science-wg/blob/master/dswg_project_resources/Project-README-template.md

[^6]: https://deepsense.ai/blog/standard-template-for-machine-learning-projects-deepsense-ais-approach/

[^7]: https://hackernoon.com/how-to-create-an-engaging-readme-for-your-data-science-project-on-github

[^8]: https://medium.datadriveninvestor.com/how-to-write-a-good-readme-for-your-data-science-project-on-github-ebb023d4a50e

[^9]: https://cubettech.com/resources/blog/the-essential-readme-file-elevating-your-project-with-a-comprehensive-document/

[^10]: https://aclanthology.org/2020.semeval-1.136.pdf

[^11]: https://git.wur.nl/bioinformatics/fte40306-advanced-machine-learning-project-data/-/blob/main/README.md

[^12]: https://github.com/lin-justin/humor
