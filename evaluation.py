import os
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import argparse

def get_prob_files_by_dataset(dataset,models):
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
    for model_name in models:
        folder = f"log/{dataset}_{model_name}_max_False_True_-6_-1"
        first_folder_with_extension = f"log/{dataset}_{model_name}_max_False_True_-6_-1/" + os.listdir(folder)[0] + "/"
        files_with_extension = [file for file in os.listdir(first_folder_with_extension) if file.endswith('logits.npy')]
        model_prob_dict[model_name] = files_with_extension
        log_paths[model_name] = first_folder_with_extension
    return model_prob_dict, log_paths


def get_acc_and_mia_acc(log_path, data_name=None):
    # read the log file
    models = ['GCN', 'GAT', 'GIN', 'GraphSage']
    acc_dict = {}
    for model in models:
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


def calculate_auc_score(prob_dict, log_paths, extra_exp):
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
            auc_score = roc_auc_score(y_true, y_prob)
            if auc_score < 0.5:
                auc_score = 1 - auc_score
            auc_score_dict[model] = auc_score
        exp_results[key] = auc_score_dict
    return exp_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, default='cora')
    parser.add_argument('--model', required=False, type=str, default="GCN")
    parser.add_argument('--extra_exp', required=False, type=bool, default=False)
    args = parser.parse_args()
    datasets = [args.dataset.lower()]
    models = [args.model.upper()]

    # datasets = ['cora', 'citeseer', 'pubmed', 'flickr']
    # datasets.remove('pubmed')
    # if not os.path.exists('auc_score.csv'):
    if True:
        df = pd.DataFrame()
        for dataset in datasets:
            # Calculate the AUC for {dataset} 
            prob_dict, log_path = get_prob_files_by_dataset(dataset, models)
            auc_score_exp_results = calculate_auc_score(prob_dict, log_path, args.extra_exp)
            print(f"{dataset}_auc_score_exp_results:\n {auc_score_exp_results} \n")
            # read dict as dataframe
            # exp_df = pd.DataFrame.from_dict(auc_score_exp_results)
            for term, value in auc_score_exp_results.items():
                for model, auc_score in value.items():
                    auc_score = auc_score if auc_score > 0.5 else 1 - auc_score
                    ddf = pd.DataFrame({"term": [term], 'dataset': [dataset], 'model': [model], 'auc_score': [auc_score]})
                    df = pd.concat([df, ddf])
        # save df to csv
        df.to_csv('auc_score.csv', index=False)
    else:
        df = pd.read_csv('auc_score.csv')
        for dataset in datasets:
            prob_dict, log_path = get_prob_files_by_dataset(dataset, models)
            # get the original accuracy and mia attack accuracy
            acc_dict = get_acc_and_mia_acc(log_path)
            print(f"{dataset}_auc_score:\n {acc_dict} \n")