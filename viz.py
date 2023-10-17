import argparse
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    # dataset_names = ['cora', 'citeseer', 'pubmed']
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,choices=['cora', 'pubmed', 'flickr', 'citeseer'], default='cora')
    parser.add_argument('--model', required=False, type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSage'], default="GCN")
    parser.add_argument('--timestr',required=False, 
                        help="Provide the time str such as 20230529_210446")
    args = parser.parse_args()
    dataset = args.dataset
    model_name =args.model
    folder = f"log/{dataset}_{model_name}_max_False_True_-6_-1"
    first_folder_with_extension = f"log/{dataset}_{model_name}_max_False_True_-6_-1/" + os.listdir(folder)[0] + "/"

    # ours_mem = np.load(first_folder_with_extension + 'new_target_pro_feat_eye_g_logits.npy')
    # ours_non_mem = np.load(
    #     first_folder_with_extension + 'target_pro_feat_eye_g_non_pro_logits.npy')
    # baseline_mem = np.load(first_folder_with_extension + 'target_feat_eye_g_logits.npy')
    # baseline_non_mem = np.load(
    #     first_folder_with_extension + 'target_feat_g_non_proa_logits.npy')
    # # Flatten tensors into one-dimensional arrays
    # data1 = ours_mem.flatten()
    # data2 = ours_non_mem.flatten()
    # data3 = baseline_mem.flatten()
    # data4 = baseline_non_mem.flatten()
    # # make above four numpy arrays into a dataframe
    # df_all = pd.DataFrame()
    # df_ours = pd.DataFrame({'ours_mem': data1})
    # # Create the plot
    # sns.kdeplot(data1, label='Target logits')
    # sns.kdeplot(data2, label='Target no proa logits')
    # sns.kdeplot(data3, label='New target logits')
    # # Set plot labels and title
    # plt.xlabel('Values')
    # plt.ylabel('Density')
    # plt.title('Distribution of Tensors')
    #
    # # Display the legend
    # plt.legend()
    #
    # if not os.path.exists(f'figure'):
    #     os.makedirs(f'figure')
    # if not os.path.exists(f'figure/{dataset}'):
    #     os.makedirs(f'figure/{dataset}')
    # # save results_dict to json file
    # plt.savefig(f'figure/{dataset}/{model_name}_dist.png', dpi=300, bbox_inches='tight')


    sns.set_style("darkgrid")
    sns.color_palette("pastel")

    folder = f"log/{dataset}_{model_name}_max_False_True_-6_-1"
    first_folder_with_extension = f"log/{dataset}_{model_name}_max_False_True_-6_-1/" + os.listdir(folder)[0]

    # path1 = r"./log"
    # model_name = r"/flickr_GAT_max_False_True_-6_-1/"
    ours_mem_file = "/new_target_pro_feat_eye_g_logits.npy"
    ours_non_file = "/target_pro_feat_eye_g_non_pro_logits.npy"
    Baseline_mem_file = "/target_feat_eye_g_logits.npy"
    Baseline_non_file = "/target_feat_g_non_proa_logits.npy"

    fig, axes = plt.subplots(1, 2)
    ours_mem = np.amax(np.load((first_folder_with_extension + ours_mem_file).replace("\\", "/")), axis=1)
    ours_non = np.amax(np.load((first_folder_with_extension + ours_non_file).replace("\\", "/")), axis=1)
    g = sns.histplot(ax=axes[0], data=[ours_mem, ours_non], bins=20,
                         kde=False)

    g.set(xticklabels=[])
    g.set(xlabel=None)
    g.set(yticklabels=[])
    g.set(ylabel=None)
    baseline_mem = np.amax(np.load((first_folder_with_extension + Baseline_mem_file).replace("\\", "/")), axis=1)
    baseline_non = np.amax(np.load((first_folder_with_extension + Baseline_non_file).replace("\\", "/")), axis=1)
    g2 = sns.histplot(ax=axes[1], data=[baseline_mem, baseline_non], bins=20,
                          kde=False)
    g2.set(xticklabels=[])
    g2.set(xlabel=None)
    g2.set(yticklabels=[])
    g2.set(ylabel=None)
    legend_labels = ['Ours (Mem)', 'Ours (Non-Mem)', 'Baseline (Mem)', 'Baseline (Non-Mem)']
    axes[0].legend(legend_labels[:2])
    axes[1].legend(legend_labels[2:])
    plt.savefig('./outputfig/' + dataset + '_' + model_name + '.png')

    # f = os.walk(path1 + model_name)
    # i = 0
    # for dirpath, dirnames, filenames in f:
    #     if len(filenames) > 0:
    #         i = i + 1
    #         print(i)
    #         if i == 1:
    #             dataset_name = 'citeseer'
    #         elif i == 2:
    #             dataset_name = 'cora'
    #         elif i == 3:
    #             dataset_name = 'flickr'
    #         print("Figure 1" + dataset_name)
    #         fig, axes = plt.subplots(1, 2)
    #         ours_mem = np.amax(np.load((dirpath + ours_mem_file).replace("\\", "/")), axis=1)
    #         ours_non = np.amax(np.load((dirpath + ours_non_file).replace("\\", "/")), axis=1)
    #         g = seaborn.histplot(ax=axes[0], data=[ours_mem, ours_non], bins=20,
    #                              kde=False)
    #
    #         g.set(xticklabels=[])
    #         g.set(xlabel=None)
    #         g.set(yticklabels=[])
    #         g.set(ylabel=None)
    #         baseline_mem = np.amax(np.load((dirpath + Baseline_mem_file).replace("\\", "/")), axis=1)
    #         baseline_non = np.amax(np.load((dirpath + Baseline_non_file).replace("\\", "/")), axis=1)
    #         g2 = seaborn.histplot(ax=axes[1], data=[baseline_mem, baseline_non], bins=20,
    #                               kde=False)
    #         g2.set(xticklabels=[])
    #         g2.set(xlabel=None)
    #         g2.set(yticklabels=[])
    #         g2.set(ylabel=None)
    #         legend_labels = ['Ours (Mem)', 'Ours (Non-Mem)', 'Baseline (Mem)', 'Baseline (Non-Mem)']
    #         axes[0].legend(legend_labels[:2])
    #         axes[1].legend(legend_labels[2:])
    #         plt.savefig('./outputfig/' + dataset_name + '.png')

    # for model in ['GCN', 'GAT', "GIN", "GraphSage"]:
    #     # Load Tensors with .npy from log files
    #     # ours_mem_file = "/new_target_pro_feat_eye_g_logits.npy"
    #     # ours_non_file = "/target_pro_feat_eye_g_non_pro_logits.npy"
    #     # Baseline_mem_file = "/targe_feat_eye_g_logits.npy"
    #     # Baseline_non_file = "/target_feat_g_non_proa_logits.npy"
    #     ours_mem = np.load(f'./exp/{model}/{dataset}_{model}_max_False_True_-6_-1/new_target_pro_feat_eye_g_logits.npy')
    #     ours_non_mem = np.load(f'./exp/{model}/{dataset}_{model}_max_False_True_-6_-1/target_pro_feat_eye_g_non_pro_logits.npy')
    #     baseline_mem = np.load(f'./exp/{model}/{dataset}_{model}_max_False_True_-6_-1/targe_feat_eye_g_logits.npy')
    #     baseline_non_mem = np.load(f'./exp/{model}/{dataset}_{model}_max_False_True_-6_-1/target_feat_g_non_proa_logits.npy')
    #     # Flatten tensors into one-dimensional arrays
    #     data1 = ours_mem.flatten()
    #     data2 = ours_non_mem.flatten()
    #     data3 = baseline_mem.flatten()
    #     data4 = baseline_non_mem.flatten()
    #     # make above four numpy arrays into a dataframe
    #     df_all = pd.DataFrame()
    #     df_ours = pd.DataFrame({'ours_mem': data1})
    #     # Create the plot
    #     sns.kdeplot(data1, label='Target logits')
    #     sns.kdeplot(data2, label='Target no proa logits')
    #     sns.kdeplot(data3, label='New target logits')
    #     # Set plot labels and title
    #     plt.xlabel('Values')
    #     plt.ylabel('Density')
    #     plt.title('Distribution of Tensors')
    #
    #     # Display the legend
    #     plt.legend()
    #
    #     if not os.path.exists(f'figure'):
    #         os.makedirs(f'figure')
    #     if not os.path.exists(f'figure/{dataset}'):
    #         os.makedirs(f'figure/{dataset}')
    #     # save results_dict to json file
    #     plt.savefig(f'figure/{dataset}/{model}_dist.png', dpi=300, bbox_inches='tight')
