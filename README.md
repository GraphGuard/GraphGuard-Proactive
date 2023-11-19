# GraphGuard: Detecting and Counteracting Training Data Misuse in Graph Neural Networks

The source code for paper: "GraphGuard: Detecting and Counteracting Training Data Misuse in Graph Neural Networks ".


## Hardware

Our artifact is compatible with common hardware settings, and supports devices with/without GPU.


## Init environment

Our experiments are conducted with PyTorch 1.12.1 and DGL. To reproduce our design, first manually install Conda, and then install other packages. 
Installing Pytorch with/without GPU might be difference (More details can be refer to https://pytorch.org/get-started/locally/).
An example for non-GPU users can setup the dependency via the following commands:

```bash
conda create --name ProactiveMIA python=3.8
conda activate ProactiveMIA
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
conda install -c dglteam dgl=1.1.0
pip install deeprobust==0.2.8
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install pandas==2.0.1
```


## Usage 

### Parameters

* target model dataset

Please specify the dataset among Cora, Citeseer, Pubmed, and Flickr for your target model training.

* target GNN architectures

Please specify the dataset among GCN, GraphSage, GAT and GIN for your target GNN model.

### Proactive Graph Detection

We integrate the whole proactive detection pipeline, including proactive graph generation, benign/data-misused GNN training, and detection.

#### Example

For running the proactive detection in Cora for GCN, you can run the comment as:

`` python Proactive_MIA_node_level_revise.py --dataset Cora --model GCN``

And results can be shown as: 

`` python evaluation.py --dataset Cora --model GCN``


### Training-graph-free Unlearning

For the unlearning pipeline, we first generate the data-misused GNN model, and then run our unlearning algorithm, including synthesising the graph, unlearning via fine-tuning, and performing MIAs to evaluate unlearning.

#### Example

For running the unlearning in Cora for GCN, you can run the comment as:
- Step 1: Generate a data-misused GNN model;
`` python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GCN``
- Step 2: Perform and evaluate our unlearning algorithm.
`` python Unlearning_node_level_revise.py --dataset Cora --model GCN``


## Baslines 

### Detection via Membership Inference Attacks

Data misuse can be detected via general Membership Inference Attacks (MIAs). We provide the results for MIAs as the baseline. 

#### Example

For running the MIAs detection in Cora for GCN, you can run the comment as:

`` python Proactive_MIA_node_level_revise.py --dataset Cora --model GCN``

### Unlearning via Retraining

Unlearning aims to generate an unlearned model training on the dataset without specific unlearning samples. Thus, the upper-bound performance of unlearning should be retraining without the unlearning samples. We provide the results for retraining as the baseline. 

#### Example

For running the retraining in Cora for GCN, you can run the comment as:
- Step 1: Generate a data-misused GNN model;
`` python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GCN``
- Step 2: Perform and evaluate our unlearning algorithm.
`` python Unlearning_retraining_node_level_revise.py --dataset Cora --model GCN``

## Reference
>
> <https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/utils.py>
> <https://github.com/ChandlerBang/Pro-GNN/blob/master/train.py>

If you have any questions, please send an email to us.
