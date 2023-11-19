import argparse
import os
import json
import numpy as np


def read_logs(log_path):
    exps = os.listdir(log_path)
    results_dict = {"For logits member": [], "For logits pro member": [], 
                   "For logits non-member": [], "For logits pro non-member": [], "Baseline MIA Acc":[]}
    for exp in exps:
        exp_path = os.path.join(log_path, exp)
        log_files = os.listdir(exp_path)
        for log_file in log_files:
            if log_file.endswith('.log'):
                log_file_path = os.path.join(exp_path, log_file)
                with open(log_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.__contains__('For logits') or line.__contains__('Baseline MIA Acc'):
                            key = line.strip().split(':')[-2].split(' - ')[-1]
                            value = line.strip().split(':')[-1]
                            results_dict[key].append(value)
                        # For logits member:0.7248322147651006
                        # For logits pro member:0.0
                        # For logits non-member:0.020594965675057208
                        # For logits pro non-member:0.0
                        # Baseline MIA Acc:0.9021739130434783
    return results_dict

def load_json(json_path):
    with open(json_path, 'r') as f:
        results_dict = json.load(f)
    return results_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', required=True, help="Please give a config.json file with param details")
    parser.add_argument('--dataset', required=True, type=str,choices=['cora', 'pubmed', 'flickr', 'citeseer'], default='cora')
    parser.add_argument('--model', required=False, type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSage'], default="GCN")
    args = parser.parse_args()
    dataset = args.dataset
    model = args.model
    folder_name = f"{dataset}_{model}_max_False_True_-6_-1"
    # if not os.path.exists(f'results/{folder_name}'):
    #     os.makedirs(f'results/{folder_name}')
    # if not os.path.exists(f'results/{folder_name}/results_dict.json'):
    #     # save results_dict to json file
    #     results_dict = read_logs(f'./log/{folder_name}')
    #     with open(f'results/{folder_name}/results_dict.json', 'w') as f:
    #         json.dump(results_dict, f)
    # else:
    #     print("File results_dict.json already exists!")
    #     results_dict = load_json(f'results/{folder_name}/results_dict.json')

    if not os.path.exists(f'results/{folder_name}'):
        os.makedirs(f'results/{folder_name}')
    # if not os.path.exists(f'results/{folder_name}/results_dict.json'):
        # save results_dict to json file
    results_dict = read_logs(f'./log/{folder_name}')
    with open(f'results/{folder_name}/results_dict.json', 'w') as f:
        json.dump(results_dict, f)
    # else:
    #     print("File results_dict.json already exists!")
    #     results_dict = load_json(f'results/{folder_name}/results_dict.json')

    # load results_dict from json file
    # print(results_dict)
    # calculate the average and standard deviation of each value in results_dict
    for key in results_dict.keys():
        results_dict[key] = [float(i) for i in results_dict[key]]
        print(f"{key}: {np.mean(results_dict[key])}+-{np.std(results_dict[key])}")