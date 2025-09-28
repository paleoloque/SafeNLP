# SafeNLP
## About
SafeNLP is a research-oriented project for detecting potentially unsafe or harmful language in question–answer pairs.
The repository provides:

+ `download_models.py`: Pre-trained models (LightGBM + Word2Vec), stored in GitHub Releases, with a pipeline to download them for inference model testing

+ `preprocessing.py`, `feature_extraction.py`: Preprocessing, feature extraction pipelines (linguistic, lexical, semantic heuristics)

+ `Safe_NLP.ipynb`: Python notebook for evaluation and demonstration

The system predicts whether a given response is safe or unsafe in terms of general public safety.

## Installation

**WARNING**: This project uses `gensim` library (`Word2Vec` word embeddings model), which is only compatible with Python 3.11 or earlier. To use the model, make sure it satisfies the requirements on your machine or clone the repository and use the tutorial notebook in the following installation steps.

For Windows (cmd):
```cmd
git clone https://github.com/paleoloque/SafeNLP.git
cd SafeNLP

py -3.11 -m venv venv-safenlp
.\venv-safenlp\Scripts\activate.bat

python -m pip install -U pip
pip install -r requirements.txt
```
For Linux (bash):
```bash
git clone https://github.com/paleoloque/SafeNLP.git
cd SafeNLP

py -3.11 -m venv venv-safenlp
source venv-safenlp/Scripts/activate

python -m pip install -U pip
pip install -r requirements.txt
```
## Reference
This project makes use of the **BeaverTails** and **Safe-RLHF** datasets:

Jincheng Ji, Tianjun Zhang, Nayeon Lee, Xinyu Wang, Yejin Shao, Hongyin Ren, Xiang Chen, George Karypis, Honglak Lee, Diyi Yang.  
*BeaverTails: Towards Safer LLMs with Contextualized Safety Evaluations.*  
arXiv:2307.04657, 2023. [https://arxiv.org/abs/2307.04657](https://arxiv.org/abs/2307.04657)

Jiaming Ji, Donghai Hong, Borong Zhang, Boyuan Chen, Josef Dai, Boren Zheng, Tianyi Qiu, Boxun Li, Yaodong Yang.  
*PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference.*  
arXiv:2406.15513, 2024. [https://arxiv.org/abs/2406.15513](https://arxiv.org/abs/2406.15513)
