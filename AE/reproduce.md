This document provides guidance to reproduce our major experimental results in our paper [GraphGuard: Detecting and Counteracting Training Data Misuse in Graph Neural Networks].

# Setup

## Hardware

This artifact requires a Linux server and can be supported (but does not require) GPU. 

## Dependency

Our experiments are conducted with PyTorch 1.12.1 and DGL. To reproduce our design, first manually install Conda, and then install other packages. 
Installing Pytorch with/without GPU might be difference (More details can be refer to https://pytorch.org/get-started/locally/).
An example for non-GPU users can setup the dependency via the following commands:

```bash
conda create --name ProactiveMIA python=3.8
conda activate ProactiveMIA
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
conda install -c dglteam dgl=1.1.0
pip install deeprobust==0.2.8 matplotlib==3.7.1 seaborn==0.12.2 pandas==2.0.1
```
## Dataset

- All four datasets, Cora, Citeseer, Pubmed, and Flickr would be automatically downloaded. 


# Reproduction Commands


## Results for E1 (Table IV)

The following command snippet includes all commands to reproduce our major results for E1, corresponding to all the entries (AUC of 4 types of GNNs x 4 datasets for both the detection of our method and baseline MIA detection) in Table IV. If your time is limited, we suggest you subsample a few GNN-dataset pairs (e.g. GCN-Cora. GCN-Citeseer, GraphSage-Cora) and only run experiments for them.
```bash
## Step 1: run our whole detection pipeline (proactive graph generation, benign/data-misused GNN training, detecting)
### the following commands produce the results in the 1st row of Table IV. 
python Proactive_MIA_node_level_revise.py --dataset Cora --model GCN
python Proactive_MIA_node_level_revise.py --dataset Cora --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Cora --model GAT
python Proactive_MIA_node_level_revise.py --dataset Cora --model GIN

### the following commands produce the results in the 2nd row of Table IV. 
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GCN
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GAT
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GIN

### the following commands produce the results in the 3rd row of Table IV. 
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GCN
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GAT
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GIN

### the following commands produce the results in the 4th row of Table IV. 
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GCN
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GAT
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GIN
```

The results for each experiment can be shown as: 

`` python evaluation.py --dataset Cora --model GCN``

## Results for E2 (Figure 4)

The following command snippet includes all commands to reproduce our major results for E2, corresponding to all the entries (output distributions of 4 types of GNNs x 4 datasets with/without our methods) in Figure 4. The output figures will be stored in /outputfig. If your time is limited, we suggest you subsample a few GNN-dataset pairs (e.g. GCN-Cora. GCN-Citeseer, GraphSage-Cora) and only run experiments for them.
```bash
## if you have run E1 and generate the files in ./log, you can directly skip to Step 2
## Step 1: run our whole detection pipeline (proactive graph generation, benign/data-misused GNN training, detecting)
python Proactive_MIA_node_level_revise.py --dataset Cora --model GCN
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GCN
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GCN
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GCN

python Proactive_MIA_node_level_revise.py --dataset Cora --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GraphSage
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GraphSage 
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GraphSage

python Proactive_MIA_node_level_revise.py --dataset Cora --model GAT
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GAT
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GAT
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GAT

python Proactive_MIA_node_level_revise.py --dataset Cora --model GIN
python Proactive_MIA_node_level_revise.py --dataset Citeseer --model GIN
python Proactive_MIA_node_level_revise.py --dataset Pubmed --model GIN
python Proactive_MIA_node_level_revise.py --dataset Flickr --model GIN

## Step 2: generate the visualisation results
### the following commands produce the figures for Cora-GCN in Figure 4
python results_processing.py --dataset Cora --model GCN
python viz.py --dataset Cora --model GCN

```

## Results for E3 (Table VII)

The following command snippet includes all commands to reproduce our major results for E3, corresponding to all the entries (MIA successful rates of 4 types of GNNs x 3 datasets for attacking the model with/without our Unlearning method) in Table VII. If your time is limited, we suggest you subsample a few GNN-dataset pairs (e.g. GCN-Cora. GCN-Citeseer, GraphSage-Cora) and only run experiments for them.

```bash
## Step 1: generate the misused models (without our unlearning design) and evaluate their privacy risk
### the following commands produce the results in the 1st column of Table VII.
python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GCN
python Unlearning_without_defence_node_level_revise.py --dataset Citeseer --model GCN
python Unlearning_without_defence_node_level_revise.py --dataset Pubmed --model GCN

### the following commands produce the results in the 4th column of Table VII.
python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GraphSage
python Unlearning_without_defence_node_level_revise.py --dataset Citeseer --model GraphSage
python Unlearning_without_defence_node_level_revise.py --dataset Pubmed --model GraphSage

### the following commands produce the results in the 7th column of Table VII.
python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GAT
python Unlearning_without_defence_node_level_revise.py --dataset Citeseer --model GAT
python Unlearning_without_defence_node_level_revise.py --dataset Pubmed --model GAT

### the following commands produce the results in the 10th column of Table VII.
python Unlearning_without_defence_node_level_revise.py --dataset Cora --model GIN
python Unlearning_without_defence_node_level_revise.py --dataset Citeseer --model GIN
python Unlearning_without_defence_node_level_revise.py --dataset Pubmed --model GIN


## Step 2: run our unlearning pipeline (synthesise the graph, unlearning via fine-tuning, performing MIAs to evaluate unlearning)
### the following commands produce the results in the 2nd column of Table VI (results for E4), and the 2nd column of Table VII.
python Unlearning_node_level_revise.py --dataset Cora --model GCN
python Unlearning_node_level_revise.py --dataset Citeseer --model GCN
python Unlearning_node_level_revise.py --dataset Pubmed --model GCN

### the following commands produce the results in the 5th column of Table VI (results for E4), and the 5th column of Table VII.
python Unlearning_node_level_revise.py --dataset Cora --model GraphSage
python Unlearning_node_level_revise.py --dataset Citeseer --model GraphSage
python Unlearning_node_level_revise.py --dataset Pubmed --model GraphSage

### the following commands produce the results in the 8th column of Table VI (results for E4), and the 8th column of Table VII.
python Unlearning_node_level_revise.py --dataset Cora --model GAT
python Unlearning_node_level_revise.py --dataset Citeseer --model GAT
python Unlearning_node_level_revise.py --dataset Pubmed --model GAT

### the following commands produce the results in the 11th column of Table VI (results for E4), and the 11th column of Table VII.
python Unlearning_node_level_revise.py --dataset Cora --model GIN
python Unlearning_node_level_revise.py --dataset Citeseer --model GIN
python Unlearning_node_level_revise.py --dataset Pubmed --model GIN

```

## Results for E4 (Table VI)

The following command snippet includes all commands to reproduce our major results for E3, corresponding to all the entries (Model Accuracy of 4 types of GNNs x 3 datasets for both the Unlearned Model via our method and baseline Retrained Model) in Table VI. If your time is limited, we suggest you subsample a few GNN-dataset pairs (e.g. GCN-Cora. GCN-Citeseer, GraphSage-Cora) and only run experiments for them.

```bash
## if you have run E3, you should have shown some results. 
## Step 1: run our unlearning pipeline (synthesise the graph, unlearning via fine-tuning, performing MIAs to evaluate unlearning)
### the following commands produce the results in the 2nd column of Table VI, and the 2nd column of Table VII (results for E3).
python Unlearning_node_level_revise.py --dataset Cora --model GCN
python Unlearning_node_level_revise.py --dataset Citeseer --model GCN
python Unlearning_node_level_revise.py --dataset Pubmed --model GCN

### the following commands produce the results in the 5th column of Table VI, and the 5th column of Table VII (results for E3).
python Unlearning_node_level_revise.py --dataset Cora --model GraphSage
python Unlearning_node_level_revise.py --dataset Citeseer --model GraphSage
python Unlearning_node_level_revise.py --dataset Pubmed --model GraphSage

### the following commands produce the results in the 8th column of Table VI, and the 8th column of Table VII (results for E3).
python Unlearning_node_level_revise.py --dataset Cora --model GAT
python Unlearning_node_level_revise.py --dataset Citeseer --model GAT
python Unlearning_node_level_revise.py --dataset Pubmed --model GAT

### the following commands produce the results in the 11th column of Table VI, and the 11th column of Table VII (results for E3).
python Unlearning_node_level_revise.py --dataset Cora --model GIN
python Unlearning_node_level_revise.py --dataset Citeseer --model GIN
python Unlearning_node_level_revise.py --dataset Pubmed --model GIN


## Step 2: run the retraining baseline (removing the unlearning nodes, retraining)
### the following commands produce the results in the 1st column of Table VI.
python Unlearning_retraining_node_level_revise.py --dataset Cora --model GCN
python Unlearning_retraining_node_level_revise.py --dataset Citeseer --model GCN
python Unlearning_retraining_node_level_revise.py --dataset Pubmed --model GCN

### the following commands produce the results in the 4th column of Table VI.
python Unlearning_retraining_node_level_revise.py --dataset Cora --model GraphSage
python Unlearning_retraining_node_level_revise.py --dataset Citeseer --model GraphSage
python Unlearning_retraining_node_level_revise.py --dataset Pubmed --model GraphSage

### the following commands produce the results in the 7th column of Table VI.
python Unlearning_retraining_node_level_revise.py --dataset Cora --model GAT
python Unlearning_retraining_node_level_revise.py --dataset Citeseer --model GAT
python Unlearning_retraining_node_level_revise.py --dataset Pubmed --model GAT

### the following commands produce the results in the 10th column of Table VI.
python Unlearning_retraining_node_level_revise.py --dataset Cora --model GIN
python Unlearning_retraining_node_level_revise.py --dataset Citeseer --model GIN
python Unlearning_retraining_node_level_revise.py --dataset Pubmed --model GIN

```
