```
Proactive-MIA-Unlearning/
├── baselines/
│   │
│   ├── nettack/
│   │   ├── __init__.py
│   │   ├── nettack.py
│   │   └── utils.py
│   │
├── config/
│   ├── citeseer.json
│   ├── cora.json
│   ├── flickr.json
│   ├── lastfm.json
│   ├── pubmed.json
│   └── reddit.json
│
├── data/
│   └── cora_target_graph_index
│
├── exp/
│   │
│   ├── GAT/
│   │   │
│   │   ├── citeseer_GAT_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_204413/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── cora_GAT_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_214939/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── flickr_GAT_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_221144/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── pubmed_GAT_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_222054/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   └── .DS_Store
│   │
│   ├── GCN/
│   │   │
│   │   ├── citeseer_GCN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_204848/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── cora_GCN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_210033/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── flickr_GCN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_220914/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── pubmed_GCN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_221532/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   ├── GIN/
│   │   │
│   │   ├── citeseer_GIN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_203439/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── cora_GIN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_211826/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── flickr_GIN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_221103/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── pubmed_GIN_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230627_115913/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   ├── GraphSage/
│   │   │
│   │   ├── citeseer_GraphSage_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_201142/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── cora_GraphSage_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_215245/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── flickr_GraphSage_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230626_221222/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
│   │   │
│   │   │
│   │   ├── pubmed_GraphSage_max_False_True_-6_-1/
│   │   │   │
│   │   │   └── 20230627_115943/
│   │   │       ├── new_nom_adj_target_logits.npy
│   │   │       ├── new_nom_adj_target_prob_list.npy
│   │   │       ├── new_target_pro_feat_eye_g_logits.npy
│   │   │       ├── new_target_pro_feat_eye_g_prob_list.npy
│   │   │       ├── results.log
│   │   │       ├── target_feat_eye_g_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_feat_eye_g_non_pro_prob_list.npy
│   │   │       ├── target_feat_eye_g_prob_list.npy
│   │   │       ├── target_feat_g_non_proa_logits.npy
│   │   │       ├── target_feat_g_non_proa_prob_list.npy
│   │   │       ├── target_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_logits.npy
│   │   │       ├── target_pro_feat_eye_g_non_pro_prob_list.npy
│   │   │       └── target_prob_list.npy
├── log/
│   │
│   ├── citeseer_GCN_max_False_True_-6_-1/
│   │   │
│   │   ├── 20230529_161446/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161507/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161528/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161548/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161610/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161630/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161650/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161710/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_161732/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   └── 20230529_162244/
│   │       ├── new_target_logits.npy
│   │       ├── results.log
│   │       ├── target_logits.npy
│   │       └── target_no_proa_logits.npy
│   │
│   │
│   ├── cora_GCN_max_False_True_-6_-1/
│   │   │
│   │   ├── 20230529_152925/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_152939/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_152952/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153008/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153020/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153041/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153101/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153112/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153123/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_153135/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230620_102720/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_103147/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_103158/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_103241/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_103307/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_114558/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_114621/
│   │   │   └── results.log
│   │   │
│   │   └── 20230629_133037/
│   │       └── results.log
│   │
│   │
│   ├── flickr_GCN_max_False_True_-6_-1/
│   │   │
│   │   ├── 20230621_101153/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230629_131755/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230629_132010/
│   │   │   └── results.log
│   │   │
│   │   └── 20230629_133537/
│   │       └── results.log
│   │
│   │
│   ├── flickr_temp/
│   │   │
│   │   ├── flickr_GAT_max_False_True_-6_-1/
│   │   │   │
│   │   │   ├── 20230620_155419/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_155648/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_155822/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_155958/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_160132/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_160321/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_160531/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_160708/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_160848/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   └── 20230620_161023/
│   │   │       ├── new_target_logits.npy
│   │   │       ├── results.log
│   │   │       ├── target_logits.npy
│   │   │       └── target_no_proa_logits.npy
│   │   │
│   │   │
│   │   ├── flickr_GCN_max_False_True_-6_-1/
│   │   │   │
│   │   │   ├── 20230620_143909/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_144204/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_144418/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_144636/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_144836/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_145056/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_145358/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_145646/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_145925/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   └── 20230620_150229/
│   │   │       ├── new_target_logits.npy
│   │   │       ├── results.log
│   │   │       ├── target_logits.npy
│   │   │       └── target_no_proa_logits.npy
│   │   │
│   │   │
│   │   ├── flickr_GIN_max_False_True_-6_-1/
│   │   │   │
│   │   │   ├── 20230620_153351/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_153539/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_153745/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_153931/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_154107/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_154251/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_154504/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_154650/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   ├── 20230620_154832/
│   │   │   │   ├── new_target_logits.npy
│   │   │   │   ├── results.log
│   │   │   │   ├── target_logits.npy
│   │   │   │   └── target_no_proa_logits.npy
│   │   │   │
│   │   │   └── 20230620_155013/
│   │   │       ├── new_target_logits.npy
│   │   │       ├── results.log
│   │   │       ├── target_logits.npy
│   │   │       └── target_no_proa_logits.npy
│   │   │
│   │   │
│   │   └── flickr_GraphSage_max_False_True_-6_-1/
│   │       │
│   │       ├── 20230621_091618/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_091827/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092014/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092200/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092350/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092531/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092740/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_092934/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       ├── 20230621_093119/
│   │       │   ├── new_target_logits.npy
│   │       │   ├── results.log
│   │       │   ├── target_logits.npy
│   │       │   └── target_no_proa_logits.npy
│   │       │
│   │       └── 20230621_093305/
│   │           ├── new_target_logits.npy
│   │           ├── results.log
│   │           ├── target_logits.npy
│   │           └── target_no_proa_logits.npy
│   │
│   │
│   │
│   ├── lastfm_GCN_max_False_True_-6_-1/
│   │   │
│   │   └── 20230620_105608/
│   │       └── results.log
│   │
│   │
│   ├── pubmed_GCN_max_False_True_-6_-1/
│   │   │
│   │   ├── 20230529_161430/
│   │   │
│   │   ├── 20230529_161800/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_162913/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_163853/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_164930/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_165920/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_171106/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_172000/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   ├── 20230529_173106/
│   │   │   ├── new_target_logits.npy
│   │   │   ├── results.log
│   │   │   ├── target_logits.npy
│   │   │   └── target_no_proa_logits.npy
│   │   │
│   │   └── 20230529_173709/
│   │       ├── new_target_logits.npy
│   │       ├── results.log
│   │       ├── target_logits.npy
│   │       └── target_no_proa_logits.npy
│   │
│   │
│   ├── reddit_GCN_max_False_True_-6_-1/
│   │   │
│   │   ├── 20230620_104439/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_104647/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_105108/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_105318/
│   │   │   └── results.log
│   │   │
│   │   ├── 20230620_114732/
│   │   │   └── results.log
│   │   │
│   │   └── 20230620_115948/
│   │       └── results.log
│   │
│   │
│   └── .DS_Store
│
├── net/
│   │
│   ├── __pycache__/
│   │   ├── estimateadj.cpython-38.pyc
│   │   ├── gat.cpython-38.pyc
│   │   ├── gcn.cpython-38.pyc
│   │   ├── gin.cpython-38.pyc
│   │   └── sage.cpython-38.pyc
│   │
│   ├── .DS_Store
│   ├── estimateadj.py
│   ├── gat.py
│   ├── gcn.py
│   ├── gin.py
│   └── sage.py
│
├── others/
│   ├── GNN-Unlearning.py
│   ├── GNN-Unlearning_total_GAE_only.py
│   ├── GNN-Unlearning_total_feature_generation.py
│   ├── GNN-Unlearning_total_test.py
│   └── Proactive-GNN.py
│
├── results/
│   │
│   ├── citeseer_GCN_max_False_True_-6_-1/
│   │   └── results_dict.json
│   │
│   ├── cora_GCN_max_False_True_-6_-1/
│   │   └── results_dict.json
│   │
│   └── .DS_Store
│
├── train/
│   │
│   ├── __pycache__/
│   │   ├── train_gnn.cpython-38.pyc
│   │   ├── train_mia.cpython-38.pyc
│   │   └── train_proactive.cpython-38.pyc
│   │
│   ├── .DS_Store
│   ├── train_gnn.py
│   ├── train_mia.py
│   └── train_proactive.py
│
├── utils/
│   │
│   ├── __pycache__/
│   │   └── graph_processing.cpython-38.pyc
│   │
│   ├── graph_processing.py
│   └── util.py
│
├── Proactive_MIA_node_level_revise.py
├── README.md
├── auc_score.csv
├── environment.txt
├── environment.yml
├── evaluation.py
├── results_processing.py
├── rptree.py
├── run_exp.sh
├── viz.py
└── viz_test.py

```
