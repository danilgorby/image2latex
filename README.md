# Im2Latex

Neural Model converting Image to Latex.

![Network structure of Im2latex Model](./imgs/model_structure.png)

As the picture shows, given an input image, a CNN and RNN encoder is applied to extract visual features firstly. And then the encoded features are used by an RNN decoder with attention mechanism to produce final formulas.



## Install Prerequsites

```
pip3 install -r requirements.txt
```



## Quick Start

We provide a small preprocessed dataset (`./sample_data/`) to check the pipeline. To train and evaluate model in this sample dataset:

```shell
python3 train.py
```



##  Train on full dataset

Download the [prebuilt dataset from Harvard](https://zenodo.org/record/56198#.V2p0KTXT6eA) and use their preprocessing scripts found [here](https://github.com/harvardnlp/im2markup).

After that, to train and evaluate model on full dataset, you can just pass the full dataset path to `train.py`:

```shel
python3 train.py --data_path=FULL_DATASET_PATH
```
