import argparse
import numpy as np
import os
import json
import time
import torch
import dgl
import copy
import logging
from train.train_gnn import Evaluation_gnn, Train_gnn_model
from train.train_mia import Baseline_mia, MIA_evaluation
from train.train_proactive import Generate_proactive_features
from utils.graph_processing import Graph_partition, Identify_proactive_nodes, Select_proactive_node, load_data, \
    normalize, subgraph_generation

# Set the seed for PyTorch
torch.manual_seed(42)
# Set the seed for NumPy
np.random.seed(42)
# Set the seed for DGL
dgl.random.seed(42)
# config logging
def config_logging(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # Create a file handler to write logs to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    # Create a stream handler to display logs on the console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(log_format)
    stream_handler.setFormatter(log_format)
    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return file_handler, stream_handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str,choices=['cora', 'pubmed', 'flickr', 'citeseer'], default='cora')
    parser.add_argument('--model', required=False, type=str, choices=['GCN', 'GAT', 'GIN', 'GraphSage'], default="GCN")
    parser.add_argument('--epochs', required=False, default=300)
    parser.add_argument('--device', required=False, type=str, default="0")
    args = parser.parse_args()
    # Get current available GPU
    device = torch.device("cuda:" + args.device) if torch.cuda.is_available() else torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     logging.info("Using MPS")
    #     device = torch.device("mps")

    # read config file
    config_file = 'config/' + args.dataset.lower() + '.json'
    with open(config_file) as f:
        config = json.load(f)
    # set up logging
    # create folder name by config params
    folder_name = 'log/' + config['params']['dataset'] + '_' + args.model + '_' + \
                  'Unlearning_without_defence' + '_' + config['params']['proattribute_method'] + '_' + config['params']['inject_only_proactive'] + '_' + \
                  config['params']['feature_inference_only'] + '_' + str(
        config['params']['proattribute_max_from']) + '_' + \
                  str(config['params']['proattribute_max_to'])
    # set up logging
    if not os.path.exists(folder_name):
        logging.info("Create folder: " + folder_name)
        os.makedirs(folder_name)
    timestamp_log_str = time.strftime("%Y%m%d_%H%M%S")
    current_log_folder = folder_name + '/' + timestamp_log_str
    os.mkdir(current_log_folder)
    log_file_name = current_log_folder + "/" + 'results.log'
    file_handler, stream_handler = config_logging(log_path=log_file_name)
    # parse config file
    params = config['params']
    # add device to params
    params['device'] = device
    dataset_name = params['dataset']
    num_classes = params['net_params']['num_labels']
    hidden_feats = params['net_params']['hidden']
    target_model_type = args.model
    proattribute_method = params['proattribute_method']
    inject_only_proactive = eval(params['inject_only_proactive'])
    feature_inference_only = eval(params['feature_inference_only'])
    # using_denoising = args.using_denoising
    # graph partition
    # num_classes = params['net_params']['num_labels']
    target_model_type = params['model']
    proattribute_method = params['proattribute_method']
    inject_only_proactive = eval(params['inject_only_proactive'])
    feature_inference_only = eval(params['feature_inference_only'])
    # load graph data
    g, features, labels, train_mask, test_mask, num_classes = load_data(dataset_name)
    # check params exist num_subsets
    if 'num_subsets' in params:
        logging.info("Start to split the graph into subsets")
        g, features, labels, train_mask, test_mask = subgraph_generation(g, features, labels, train_mask, test_mask,
                                                                         params)
    else:
        logging.info("Do not need to split the graph into subsets")
    # normalize features
    target_g, target_features, target_labels, target_train_mask, target_test_mask, \
        shadow_g, shadow_features, shadow_labels, shadow_train_mask, shadow_test_mask = \
        Graph_partition(g, features, labels, train_mask, test_mask)
    # select a proper proactive feature and label index
    logging.info("Select the proactive feature and label index")
    proactive_features_index, proactive_label = Select_proactive_node(target_features, target_labels)

    # select the proactive nodes
    logging.info("Identify the target proactive nodes")
    proactive_node_index_target = Identify_proactive_nodes(target_features, target_labels, proactive_features_index,
                                                           proactive_label)
    logging.info('Total ' + str(len(target_labels)) + ' nodes in target set')
    logging.info('Generate ' + str(proactive_node_index_target.size()) + ' proactive node in target set')
    # print(target_features.size())
    logging.info("Identify the shadow proactive nodes")
    proactive_node_index_shadow = Identify_proactive_nodes(shadow_features, shadow_labels, proactive_features_index,
                                                           proactive_label)
    logging.info('Total ' + str(len(shadow_labels)) + ' nodes in shadow set')
    logging.info('Generate ' + str(proactive_node_index_shadow.size()) + ' proactive node in shadow set')
    # evaluate the MIA on original proactive nodes
    target_evaluation_mask = torch.zeros([target_features.size()[0], 1]).bool()
    shadow_evaluation_mask = torch.zeros([shadow_features.size()[0], 1]).bool()
    target_evaluation_mask[proactive_node_index_target, 0] = True
    shadow_evaluation_mask[proactive_node_index_shadow, 0] = True

    # Train: traget_model (target feature (without proactive node index)) vs new_target_model (proactive feature)
    # eval: target_model (proactive feature) -> 0-nonmember vs new_target_model (proactive feature) -> 1-member

    proactive_nodex_index_target_mask = torch.zeros(target_g.number_of_nodes(), dtype=torch.bool)
    proactive_nodex_index_target_mask[proactive_node_index_target] = True
    # train target GNN model
    logging.info("Start train the target GNN model")

    if dataset_name == 'citeseer':
        params['epochs']=300

    target_model = Train_gnn_model(params, target_g,
                                   target_features,
                                   target_labels,
                                   target_train_mask,
                                   target_test_mask)

    target_eval_acc, target_logits, target_prob_list = Evaluation_gnn(target_model, target_g, target_features,
                                                                      target_labels)
    logging.info(f"The evaluation accuracy of the target model is:{target_eval_acc}")
    # inti an eyes graph based on target graph
    eyes_g = dgl.DGLGraph()
    eyes_g.add_nodes(target_g.number_of_nodes())
    eyes_g.add_edges(torch.tensor(range(target_g.number_of_nodes())), torch.tensor(range(target_g.number_of_nodes())))
    target_eval_acc, targe_feat_eye_g_logits, targe_feat_eye_g_prob_list = Evaluation_gnn(target_model, eyes_g,
                                                                                          target_features,
                                                                                          target_labels)

    # train shadow GNN model
    logging.info("Start train the shadow GNN model")
    shadow_model = Train_gnn_model(params, shadow_g,
                                   shadow_features,
                                   shadow_labels,
                                   shadow_train_mask,
                                   shadow_test_mask)
    shadow_eval_acc, shadow_logits, _ = Evaluation_gnn(shadow_model, shadow_g, shadow_features, shadow_labels)
    logging.info(f"The evaluation accuracy of the shadow model is:{shadow_eval_acc}")
    # create masked graph, features, labels
    target_g_masked = target_g.subgraph(proactive_nodex_index_target_mask)
    target_features_masked = target_features[proactive_nodex_index_target_mask]
    target_labels_masked = target_labels[proactive_nodex_index_target_mask]
    logging.info("Start train the target GNN model without proactive node index")
    target_model_no_proa_nodeidx = Train_gnn_model(params, target_g_masked,
                                                   target_features_masked,
                                                   target_labels_masked,
                                                   target_train_mask,
                                                   target_test_mask)
    # evalue the target model without proactive node index
    # TODO replace target_g_masked with eyes_g
    target_acc_no_proa_nodeidx, target_feat_g_non_proa_logits, target_feat_g_non_proa_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        target_g,
        target_features,
        target_labels)
    logging.info(
        f"The evaluation accuracy of the target model without proactive node index is:{target_acc_no_proa_nodeidx}")
    # TODO: save the target_logits_non_pro , change target_g to eyes_g
    _, target_feat_eye_g_non_pro_logits, target_feat_eye_g_non_pro_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        eyes_g, target_features,
        target_labels)

    logging.info("Generate the proactive features")
    proactive_target_features, proactive_attribute_trigger_index = Generate_proactive_features(proattribute_method,
                                                                                               target_model,
                                                                                               proactive_node_index_target,
                                                                                               target_g,
                                                                                               target_features,
                                                                                               target_labels)
    # train GNN model and attack (with proactive)
    logging.info("Start train the New Target GNN model with proactive feature:")
    new_target_model = Train_gnn_model(params, target_g,
                                       proactive_target_features, target_labels,
                                       target_train_mask, target_test_mask)
    # TODO: change the target_g to eyes_g
    new_target_eval_acc, new_target_pro_feat_eye_g_logits, new_target_pro_feat_eye_g_prob_list = Evaluation_gnn(
        new_target_model,
        eyes_g,
        proactive_target_features,
        target_labels)
    logging.info(f"The evaluation accuracy of the new target model is:{new_target_eval_acc}")
    _, target_pro_feat_eye_g_non_pro_logits, target_pro_feat_eye_g_non_pro_prob_list = Evaluation_gnn(
        target_model_no_proa_nodeidx,
        eyes_g, proactive_target_features,
        target_labels)


    # Now save all the logits and labels
    # Save true label probability
    np.save(f'{current_log_folder}/target_prob_list.npy', target_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_prob_list.npy', targe_feat_eye_g_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_g_non_proa_prob_list.npy',
            target_feat_g_non_proa_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_non_pro_prob_list.npy',
            target_feat_eye_g_non_pro_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/new_target_pro_feat_eye_g_prob_list.npy',
            new_target_pro_feat_eye_g_prob_list.detach().numpy())
    np.save(f'{current_log_folder}/target_pro_feat_eye_g_non_pro_prob_list.npy',
            target_pro_feat_eye_g_non_pro_prob_list.detach().numpy())
    # save target, target_no_proa logit and new target logit
    np.save(f'{current_log_folder}/target_logits.npy', target_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_logits.npy', targe_feat_eye_g_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_g_non_proa_logits.npy', target_feat_g_non_proa_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_feat_eye_g_non_pro_logits.npy',
            target_feat_eye_g_non_pro_logits.detach().numpy())
    np.save(f'{current_log_folder}/new_target_pro_feat_eye_g_logits.npy',
            new_target_pro_feat_eye_g_logits.detach().numpy())
    np.save(f'{current_log_folder}/target_pro_feat_eye_g_non_pro_logits.npy',
            target_pro_feat_eye_g_non_pro_logits.detach().numpy())

    # new evaluation the proactive MIA
    ## use original graph
    try:
        logits_mem = target_model(target_g.adjacency_matrix(), proactive_target_features)
        logits_pro_mem = new_target_model(target_g.adjacency_matrix(), proactive_target_features)
    except:
        logits_mem = target_model(target_g, proactive_target_features)
        logits_pro_mem = new_target_model(target_g, proactive_target_features)
    logging.info("=============Final Results=============")
    logging.info("============Trigger Injection (second value high means injected)=============")
    _, indices = torch.max(logits_mem[proactive_node_index_target], dim=1)
    labels = copy.deepcopy(indices)
    labels = torch.where((labels != -1), proactive_label, labels)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits member:{correct.item() * 1.0 / len(labels)}")
    _, indices = torch.max(logits_pro_mem[proactive_node_index_target], dim=1)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits pro member:{correct.item() * 1.0 / len(labels)}")
    ## use shadow graph
    # shadow graph features
    shadow_graph_proactive_features = copy.deepcopy(shadow_features)
    if inject_only_proactive:
        for i in proactive_node_index_shadow.numpy():
            for j in proactive_attribute_trigger_index.numpy():
                with torch.no_grad():
                    shadow_graph_proactive_features[i][j] = 1
    else:
        for i in range(len(shadow_graph_proactive_features)):
            for j in proactive_attribute_trigger_index.numpy():
                with torch.no_grad():
                    shadow_graph_proactive_features[i][j] = 1
    # normalization
    shadow_graph_proactive_features[shadow_graph_proactive_features != 0] = 1
    target_graph_proactive_features = normalize(shadow_graph_proactive_features)
    if feature_inference_only:
        src_idx = torch.tensor(range(len(shadow_graph_proactive_features)), dtype=torch.int64)
        shadow_g = dgl.DGLGraph()
        shadow_g.add_nodes(len(shadow_graph_proactive_features))
        shadow_g.add_edges(src_idx, src_idx)
    try:
        logits_non_mem = target_model(shadow_g.adjacency_matrix(), shadow_graph_proactive_features)
        logits_pro_non_mem = new_target_model(shadow_g.adjacency_matrix(), shadow_graph_proactive_features)
    except:
        logits_non_mem = target_model(shadow_g, shadow_graph_proactive_features)
        logits_pro_non_mem = new_target_model(shadow_g, shadow_graph_proactive_features)
    if inject_only_proactive:
        _, indices = torch.max(logits_non_mem[proactive_node_index_shadow], dim=1)
    else:
        _, indices = torch.max(logits_non_mem, dim=1)

    labels = copy.deepcopy(indices)
    labels = torch.where((labels != -1), proactive_label, labels)
    # print(labels)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits non-member:{correct.item() * 1.0 / len(labels)}")
    if inject_only_proactive:
        _, indices = torch.max(logits_pro_non_mem[proactive_node_index_shadow], dim=1)
    else:
        _, indices = torch.max(logits_pro_non_mem, dim=1)
    correct = torch.sum(indices == labels)
    logging.info(f"For logits pro non-member:{correct.item() * 1.0 / len(labels)}")
    # target_mode vs new_target_model
    # target_model(small value) vs new_target_model(large value) -> threshold

    # # evaluate the proactive MIA
    # Evaluation_proactive_mia(target_model, new_target_model, shadow_model,
    #                          target_g, target_features,
    #                          # shadow_g, shadow_features,
    #                          target_g, proactive_target_features,
    #                          proactive_node_index_target, # proactive_node_index_shadow,
    #                          target_labels, proactive_label)

    attack_model = Baseline_mia(params, target_g, shadow_g, shadow_model, target_features, shadow_features)
    attack_acc = MIA_evaluation(attack_model, target_model,
                                target_g, target_features, shadow_g, shadow_features, target_evaluation_mask,
                                shadow_evaluation_mask)
    logging.info(f"Baseline MIA Acc:{attack_acc}")
    # close the log file handler
    logging.info("============End=============")
    stream_handler.close()
    file_handler.close()
    print("======Reproduce Results  E3==========")
    print("The Original MIA Successul Rate is:")
    print(attack_acc)
