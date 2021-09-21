# GTR
Code and data for our paper [Retrieving Complex Tables with Multi-Granular Graph Representation Learning](https://arxiv.org/abs/2105.01736) at SIGIR 2021.

## Quick Links
  - [Preliminary](#preliminary)
  - [Run](#run)
  - [Citation](#citation)

## Preliminary

[Install DGL](https://docs.dgl.ai/en/0.4.x/install/):
```bash
conda install -c dglteam dgl-cuda10.2  # pay attention to the cuda version
```

[Install fastText](https://fasttext.cc/docs/en/support.html):
```bash
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
```

Download [pretrained word vectors](https://fasttext.cc/docs/en/pretrained-vectors.html):
```bash
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
unzip wiki.en.zip
```

Install [trec_eval tool](https://github.com/usnistgov/trec_eval):
```bash
git clone https://github.com/usnistgov/trec_eval.git
cd trec_eval
make
```

Install other requirements:
```bash
pip install -r requirements.txt
```

## Run
To run cross validation on WikiTables dataset:
```bash
python run.py --exp cross_validation --config configs/wikitables.json
```

## Citation
If you use our code in your research, please cite our work:
```bibtex
@article{wang2021retrieving,
  title={Retrieving Complex Tables with Multi-Granular Graph Representation Learning},
  author = {Wang, Fei and Sun, Kexuan and Chen, Muhao and Pujara, Jay and Szekely, Pedro},
  journal={Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```