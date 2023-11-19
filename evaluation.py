import os
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

def get_prob_files_by_dataset(dataset, model_name):
    # models = ['GCN', 'GAT', 'GIN', 'GraphSage']
    model_prob_dict = {}
    log_paths = {}
    # models = [model]
    # for model_name in models:
    #     folder = f"log/{dataset}_{model_name}_max_False_True_-6_-1"
    #     files_with_extension = [file for file in os.listdir(folder) if file.endswith('logits.npy')]
    #     model_prob_dict[model_name] = files_with_extension
    #     log_paths[model_name] = folder
    # return model_prob_dict, log_paths
    # for model_name in models:
    # Unlearning_without_defence_
    # folder = f"log/{dataset}_{model_name}_Unlearning_without_defence_max_False_True_-6_-1"
    folder = f"log/{dataset}_{model_name}_max_False_True_-6_-1"
    first_folder_with_extension = f"log/{dataset}_{model_name}_max_False_True_-6_-1/" + os.listdir(folder)[-1] + "/"
    print(first_folder_with_extension)
    files_with_extension = [file for file in os.listdir(first_folder_with_extension) if file.endswith('logits.npy')]
    model_prob_dict[model_name] = files_with_extension
    log_paths[model_name] = first_folder_with_extension
    return model_prob_dict, log_paths


def get_acc_and_mia_acc(log_path, model):
    # read the log file
    acc_dict = {}
    log_file = os.path.join(log_path[model], 'results.log')
    with open(log_file, 'r') as f:
        lines = f.readlines()
    # get the original accuracy and mia attack accuracy
    acc_index = 12 if len(lines) == 77 else 18
    new_acc_index = 61 if len(lines) == 77 else 67
    mia_acc_index = 75 if len(lines) == 77 else 81
    acc = float(lines[acc_index].split(':')[-1])
    new_acc = float(lines[new_acc_index].split(':')[-1])
    mia_acc = float(lines[mia_acc_index].split(':')[-1])
    acc_dict[model] = [acc, new_acc, mia_acc]
    return acc_dict

def calculate_fp_tp(true_labels, predicted_labels, positive_class):
    """
    Calculate False Positives (FP) and True Positives (TP) given true labels and predicted labels.
    
    Args:
    true_labels (list or numpy array): True labels (ground truth).
    predicted_labels (list or numpy array): Predicted labels.
    positive_class: The class label considered as the positive class.
    
    Returns:
    fp (int): Number of False Positives.
    tp (int): Number of True Positives.
    """
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    
    # Determine which instances are positive and negative
    is_positive = true_labels == positive_class
    
    # Calculate True Positives (TP) and False Positives (FP)
    tp = np.sum((predicted_labels == positive_class) & is_positive)
    fp = np.sum((predicted_labels == positive_class) & ~is_positive)
    fn = np.sum((predicted_labels != positive_class) & is_positive)
    tn = np.sum((predicted_labels != positive_class) & ~is_positive)
    return fp/(fp+tn), tp/(tp+fn)

def calculate_auc_score(prob_dict, log_paths, extra_exp):
    # ours = ['target_pro_feat_eye_g_non_pro_logits.npy', 'new_target_pro_feat_eye_g_logits.npy']
    # baseline = ['target_feat_g_non_proa_logits.npy', 'target_logits.npy']
    if dataset == 'citeseer':
        baseline = ['target_non_mem_logits.npy', 'target_mem_logits.npy']
        ours = ['target_pro_non_mem_logits.npy', 'target_pro_mem_logits.npy']
    else:
        ours = ['target_pro_feat_eye_g_non_pro_logits.npy', 'new_target_pro_feat_eye_g_logits.npy']
        baseline = ['target_feat_g_non_proa_logits.npy', 'target_logits.npy']
    compeff = ['target_feat_eye_g_non_pro_logits.npy', 'target_feat_eye_g_logits.npy']
    comprobust = ['target_pro_feat_eye_g_non_pro_logits.npy', 'new_nom_adj_target_logits.npy']
    # get key and prob from dict
    if extra_exp:
        file_pairs = {'ours': ours, 'baseline': baseline, 'compeff': compeff, 'comprobust': comprobust}
    else:
        file_pairs = {'ours': ours, 'baseline': baseline}
    exp_results = {}
    for key, file_pair in file_pairs.items():
        auc_score_dict = {}
        for model, _ in prob_dict.items():
            # read prob from npy file
            file_path_0 = os.path.join(log_paths[model], file_pair[0])
            file_path_1 = os.path.join(log_paths[model], file_pair[1])
            logits_0 = np.load(file_path_0)
            logits_1 = np.load(file_path_1)
            # logits to prob using torch.nn.functional.softmax
            y_prob_0 = F.softmax(torch.from_numpy(logits_0), dim=1)
            y_prob_1 = F.softmax(torch.from_numpy(logits_1), dim=1)
            y_true_0 = np.array([0] * len(y_prob_0))
            y_true_1 = np.array([1] * len(y_prob_1))
            # combind y_prob and y_true
            y_prob = np.concatenate((y_prob_0.max(1).values, y_prob_1.max(1).values), axis=0)
            y_true = np.concatenate((y_true_0, y_true_1), axis=0)
            # calculate FP and TP
            # get pred label by y_prob
            threshold_list = np.arange(0, 1, 0.001)
            fpr_tpr_dict = {'results':[]}
            for threshold in threshold_list:
                predicted_labels = np.where(y_prob > threshold, 1, 0)
                fpr, tpr = calculate_fp_tp(y_true, predicted_labels, 1)
                fpr_tpr_dict['results'].append({'threshold': threshold, 'fpr': fpr, 'tpr': tpr})
            # calculate FP Rate and TP rate
            # fp_rate = fp / len(y_true_0)
            # tp_rate = tp / len(y_true_1)
            # print(f"{model}_{key}_fpr_tpr_dict:\n {fpr_tpr_dict} \n")
            # save fpr_tpr_dict to csv
            df = pd.DataFrame(fpr_tpr_dict['results'])
            df.to_csv(f'{model}_{key}_fpr_tpr_dict.csv', index=False)
            auc_score = roc_auc_score(y_true, y_prob)
            if auc_score < 0.5:
                auc_score = 1 - auc_score
            auc_score_dict[model] = auc_score
            # add the trp and fpr to dict
            auc_score_dict[model] = {'auc_score':auc_score, 'tpr': tpr, 'fpr': fpr}
        exp_results[key] = auc_score_dict
    return exp_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,choices=['cora', 'pubmed', 'flickr', 'citeseer'], default='cora')
    parser.add_argument('--model', required=False, type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSage'], default="GCN")
    parser.add_argument('--extra_exp', required=False, type=bool, default=False)
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    df = pd.DataFrame()
    # Calculate the AUC for {dataset} 
    prob_dict, log_path = get_prob_files_by_dataset(dataset, model)
    auc_score_exp_results = calculate_auc_score(prob_dict, log_path, args.extra_exp)
    print(f"{dataset}_auc_score_exp_results:\n {auc_score_exp_results} \n")
    # read dict as dataframe
    # exp_df = pd.DataFrame.from_dict(auc_score_exp_results)
    for term, value in auc_score_exp_results.items():
        for model, scores in value.items():
            auc_score = scores['auc_score']
            # tpr_score = scores['tpr']
            # fpr_score = scores['fpr']
            auc_score = auc_score if auc_score > 0.5 else 1 - auc_score
            ddf = pd.DataFrame({"term": [term], 'dataset': [dataset], 'model': [model], 'auc_score': [auc_score]})
            df = pd.concat([df, ddf])
    # save df to csv
    df.to_csv(f'{dataset}_{model}_auc_score.csv', index=False)