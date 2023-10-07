import numpy as np
import os
import glob
import seaborn
import matplotlib
import matplotlib.pyplot as plt

# penguins = seaborn.load_dataset("penguins")
# print(type(penguins))
# plt.show()

# path1 = r"./data/Proactive_MIA_Unlearning"
# model_name = r"/GCN"
# ours_mem_file = "/new_target_pro_feat_eye_g_prob_list.npy"
# ours_non_file = "/target_pro_feat_eye_g_non_pro_prob_list.npy"
# Baseline_mem_file = "/target_feat_eye_g_prob_list.npy"
# Baseline_non_file = "/target_feat_g_non_proa_prob_list.npy"
#
# f = os.walk(path1+model_name)
# i = 0
# for dirpath,dirnames,filenames in f:
#     if len(filenames) > 0:
#         print("Figure 1" + dirpath)
#
#         ours_mem = np.load((dirpath+ours_mem_file).replace("\\", "/"))
#         ours_non = np.load((dirpath+ours_non_file).replace("\\", "/"))
#         seaborn.displot([ours_mem, ours_non], bins=20,
#                         color=['red', 'skyblue'],
#                         kde=False)  # , x="flipper_length_mm", hue="species")
#         plt.show()
#
#         print("Figure 2" + dirpath)
#
#         baseline_mem = np.load((dirpath + Baseline_mem_file).replace("\\", "/"))
#         baseline_non = np.load((dirpath + Baseline_non_file).replace("\\", "/"))
#         seaborn.displot([baseline_mem, baseline_non], bins=20,
#                         color=['red', 'skyblue'],
#                         kde=False)
#         # matplotlib.hold(True)
#         # seaborn.displot(max_value_twod_row_2, bins=np.arange(0.0, 1.0, 0.1), color='skyblue',
#         #                 kde=False)  # , x="flipper_length_mm", hue="species")
#         plt.show()

seaborn.set_style("darkgrid")
seaborn.color_palette("pastel")

path1 = r"./exp"
model_name = r"/GAT/flickr_GAT_max_False_True_-6_-1/"
ours_mem_file = "/new_target_pro_feat_eye_g_logits.npy"
ours_non_file = "/target_pro_feat_eye_g_non_pro_logits.npy"
Baseline_mem_file = "/targe_feat_eye_g_logits.npy"
Baseline_non_file = "/target_feat_g_non_proa_logits.npy"

f = os.walk(path1+model_name)
i = 0
for dirpath,dirnames,filenames in f:
    if len(filenames) > 0:
        i = i + 1
        print(i)
        if i == 1:
            dataset_name = 'citeseer'
        elif i == 2:
            dataset_name = 'cora'
        elif i == 3:
            dataset_name = 'flickr'
        print("Figure 1" + dataset_name)
        fig, axes = plt.subplots(1, 2)
        ours_mem = np.amax(np.load((dirpath+ours_mem_file).replace("\\", "/")), axis=1)
        ours_non = np.amax(np.load((dirpath+ours_non_file).replace("\\", "/")), axis=1)
        g = seaborn.histplot(ax=axes[0],data=[ours_mem, ours_non], bins=20,
                        kde=False) 
        
        g.set(xticklabels=[])
        g.set(xlabel=None)
        g.set(yticklabels=[])
        g.set(ylabel=None)
        baseline_mem = np.amax(np.load((dirpath + Baseline_mem_file).replace("\\", "/")), axis=1)
        baseline_non = np.amax(np.load((dirpath + Baseline_non_file).replace("\\", "/")), axis=1)
        g2 = seaborn.histplot(ax=axes[1],data=[baseline_mem, baseline_non], bins=20,
                        kde=False)
        g2.set(xticklabels=[])
        g2.set(xlabel=None)
        g2.set(yticklabels=[])
        g2.set(ylabel=None)
        legend_labels = ['Ours (Mem)', 'Ours (Non-Mem)', 'Baseline (Mem)', 'Baseline (Non-Mem)']
        axes[0].legend(legend_labels[:2])
        axes[1].legend(legend_labels[2:])
        plt.savefig('./outputfig/' + dataset_name + '.png')
    