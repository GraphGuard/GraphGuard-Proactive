import argparse
import os
import json
import time
import torch
import dgl
import logging
from train.train_gnn import Evaluation_gnn, Train_gnn_model
from utils.graph_processing import Graph_partition, Identify_proactive_nodes, Select_proactive_node, load_data, \
    subgraph_generation

# Set the seed for PyTorch
torch.manual_seed(42)
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
    parser.add_argument('--dataset', required=False, default="cora")
    parser.add_argument('--epochs', required=False, default=300)
    parser.add_argument('--model', required=False, type=str, default="GCN")
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

    # remove proactive nodes

    remain_node_index_target = list(range(target_g.number_of_nodes()))
    for pro_index in proactive_node_index_target:
        remain_node_index_target.remove(pro_index)

    new_target_g = dgl.remove_nodes(target_g, torch.tensor(proactive_node_index_target))
    new_target_features = torch.index_select(target_features, 0, torch.LongTensor(remain_node_index_target))
    new_target_labels = torch.index_select(target_labels, 0, torch.LongTensor(remain_node_index_target))
    new_target_train_mask = torch.index_select(target_train_mask, 0, torch.LongTensor(remain_node_index_target))
    new_target_test_mask = torch.index_select(target_test_mask, 0, torch.LongTensor(remain_node_index_target))


    # train target GNN model
    logging.info("Start train the target GNN model")
    # breakpoint()
    new_target_model = Train_gnn_model(params, new_target_g,
                                   new_target_features,
                                   new_target_labels,
                                   new_target_train_mask,
                                   new_target_test_mask)

    new_target_eval_acc, new_target_logits, new_target_prob_list = Evaluation_gnn(new_target_model, new_target_g, new_target_features,
                                                                      new_target_labels)
    logging.info(f"The evaluation accuracy of the target model is:{new_target_eval_acc}")

    # close the log file handler
    logging.info("============End=============")
    stream_handler.close()
    file_handler.close()
    print("======Reproduce Results  E4==========")
    print("The Retrained GNN Accuracy is:")
    print(new_target_eval_acc)
