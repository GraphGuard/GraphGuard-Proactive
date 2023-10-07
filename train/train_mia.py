import torch
import logging
import torch.nn.functional as F
from net.gcn import Attack
def Baseline_mia(params, target_g, shadow_g, shadow_model, target_features, shadow_features):
    """
    This function is for implement a baseline MIA. It will print out the attack successful rate.

    Inputs:
    A shadow dataset including:
    shadow_g:
    shadow_features:
    shadow_labels:
    shadow_train_mask:
    shadow_test_mask:

    Outputs:
    attack_model
    """
    # shadow graph partitions
    # target_g, target_features, target_labels, target_train_mask, target_test_mask, \
    #     shadow_g, shadow_features, shadow_labels, shadow_train_mask, shadow_test_mask = \
    #     Graph_partition(g, features, labels, train_mask, test_mask)
    # # train shadow model
    # print("The shadow GNN model accuracy for baseline attack is:")
    # shadow_model = Train_gnn_model('GCN', target_g, target_features, target_labels, target_train_mask, target_test_mask)



    # train attack model
    shadow_model.eval()

    # # check the train-test gap of model:
    # print("Train test gap")
    # Evaluation_gnn(shadow_model, target_g, target_features, target_labels)
    # Evaluation_gnn(shadow_model, shadow_g, shadow_features, shadow_labels)

    logits_target = shadow_model(target_g.adjacency_matrix(), target_features)
    logits_shadow = shadow_model(shadow_g.adjacency_matrix(), shadow_features)
    logits_total = torch.vstack((logits_target, logits_shadow))

    member_labels = torch.ones([target_features.size()[0], 1]).long()
    nonmember_labels = torch.zeros([shadow_features.size()[0], 1]).long()
    logits_labels = torch.vstack((member_labels, nonmember_labels)).squeeze(1)

    # get params
    num_classes = params['net_params']['num_labels']
    device = params['device']
    attack_model = Attack(int(num_classes))
    attack_model = attack_model.to(device)
    opt_attack = torch.optim.Adam(attack_model.parameters())
    logits_total = logits_total.to(device)
    logits_labels = logits_labels.to(device)
    attack_model.train()
    for epoch in range(500):
        total = 0
        correct = 0
        attack_logits = attack_model(logits_total)
        attack_loss = F.cross_entropy(attack_logits, logits_labels)
        # print(attack_loss)
        opt_attack.zero_grad()
        attack_loss.backward(retain_graph=True)
        opt_attack.step()
        _, indices = torch.max(attack_logits, dim=1)
        correct = torch.sum(indices == logits_labels)
        total += len(logits_labels)
        if (epoch + 1) % 100 == 0:
            logging.info(f"Epoch: {epoch+1}, Attack loss: {attack_loss}, Attack accuracy: {correct.item() * 1.0 / total}")
    return attack_model

def MIA_evaluation(attack_model, target_model,
                   target_g, target_features, shadow_g, shadow_features, target_evaluation_mask, shadow_evaluation_mask):
    """
    This function is for evaluate the performance of MIA using attack model. It will print our the attack performance.

    Inputs:
    attack_model: the generated attack model for MIA.
    target_model: the target model.
    Two member and nonmember graph containing:
    g: graph in DGL graph format
    features: features in Tensor format

    evaluation_index: The nodes that we evaluate their MIA.

    Outputs:
    None
    """
    attack_model = attack_model.to('cpu')
    target_model = target_model.to('cpu')
    attack_model.eval()
    target_model.eval()
    logits_target = target_model(target_g.adjacency_matrix(), target_features)
    # print(torch.mean(logits_target))
    # Evaluation_gnn(target_model, target_g, target_features, target_labels)
    logits_shadow = target_model(shadow_g.adjacency_matrix(), shadow_features)
    # print(torch.mean(logits_shadow))
    # Evaluation_gnn(target_model, shadow_g, shadow_features, shadow_labels)

    logits_total = torch.vstack((logits_target, logits_shadow))

    member_labels = torch.ones([target_features.size()[0], 1]).long()
    nonmember_labels = torch.zeros([shadow_features.size()[0], 1]).long()

    # print(member_labels.size())

    logits_labels = torch.vstack((member_labels, nonmember_labels)).squeeze(1)

    # print(target_evaluation_mask.size())
    # mask_total = torch.vstack((target_evaluation_mask, shadow_evaluation_mask)).squeeze(1)
    attack_logits_total = attack_model(logits_total)

    _, indices = torch.max(attack_logits_total, dim=1)
    # print(indices.size())
    # print(logits_labels.size())
    # print(mask_total.size())
    # print((indices[mask_total] == logits_labels[mask_total]).size())
    if len(target_evaluation_mask.shape) == 2:
        mask_total = torch.vstack((target_evaluation_mask, shadow_evaluation_mask))
    else:
        mask_total = torch.vstack((target_evaluation_mask.unsqueeze(1), shadow_evaluation_mask.unsqueeze(1)))
    if len(mask_total.shape) == 2:
        mask_total = mask_total.squeeze(1)
    else:
        print("2:1 ", mask_total.shape)
    correct = torch.sum((indices[mask_total] == logits_labels[mask_total]))
    return correct.item() * 1.0 / len(logits_labels[mask_total])

def Unlearning_MIA_evaluation(attack_model, target_model,
                   target_g, target_features, target_evaluation_mask):
    """
    This function is for evaluate the performance of MIA for unlearned samples using attack model. It will print our the attack performance.

    Inputs:
    attack_model: the generated attack model for MIA.
    target_model: the target model.
    target graph containing:
    g: graph in DGL graph format
    features: features in Tensor format

    evaluation_index: The nodes that we evaluate their MIA.

    Outputs:
    None
    """
    attack_model = attack_model.to('cpu')
    target_model = target_model.to('cpu')
    attack_model.eval()
    target_model.eval()

    labels_target = torch.ones([target_features.size()[0], 1]).long()
    labels_unlearned = torch.ones([target_features.size()[0], 1]).long()
    
    with torch.no_grad():
        logits = attack_model(target_model(target_g.adjacency_matrix(), target_features))
        logits = logits[target_evaluation_mask]
        labels_t = labels_target[target_evaluation_mask]
        labels_u = labels_unlearned[target_evaluation_mask]
        _, indices = torch.max(logits, dim=1)
        # correct_t = torch.sum(indices == labels_t)
        # correct_u = torch.sum(indices == labels_u)
        # print("======log============")
        # print(indices)
        # print(len(indices))
        # print(correct_u)
        # print(len(correct_u))
        # print(correct_t)
        # print(len(correct_t))
        correct = torch.sum(indices)
        return correct.item() / len(indices)
        # return (correct_t.item() * 0.5 + correct_u.item() * 0.5) / len(indices)

def additional_Unlearning_MIA_evaluation(attack_model, target_model,
                   target_g, target_features, target_evaluation_mask):
    """
    This function is for evaluate the performance of MIA for unlearned samples using attack model. It will print our the attack performance.

    Inputs:
    attack_model: the generated attack model for MIA.
    target_model: the target model.
    target graph containing:
    g: graph in DGL graph format
    features: features in Tensor format

    evaluation_index: The nodes that we evaluate their MIA.

    Outputs:
    None
    """
    attack_model = attack_model.to('cpu')
    target_model = target_model.to('cpu')
    attack_model.eval()
    target_model.eval()
    logits_total = target_model(target_g.adjacency_matrix(), target_features)
    # print(torch.mean(logits_target))
    # Evaluation_gnn(target_model, target_g, target_features, target_labels)
    # logits_shadow = target_model(shadow_g.adjacency_matrix(), shadow_features)
    # print(torch.mean(logits_shadow))
    # Evaluation_gnn(target_model, shadow_g, shadow_features, shadow_labels)

    # logits_total = torch.vstack((logits_target, logits_shadow)).squeeze(1)

    logits_labels = torch.ones([target_features.size()[0], 1]).long()
    # nonmember_labels = torch.zeros([shadow_features.size()[0], 1]).long()

    # print(member_labels.size())

    # logits_labels = torch.vstack((member_labels, nonmember_labels)).squeeze(1)

    # print(target_evaluation_mask.size())
    # mask_total = torch.vstack((target_evaluation_mask, shadow_evaluation_mask)).squeeze(1)
    attack_logits_total = attack_model(logits_total)

    _, indices = torch.max(attack_logits_total, dim=1)
    # print(indices.size())
    # print(logits_labels.size())
    # print(mask_total.size())
    # print((indices[mask_total] == logits_labels[mask_total]).size())
    if len(target_evaluation_mask.shape) == 2:
        mask_total = target_evaluation_mask
    else:
        mask_total = target_evaluation_mask.unsqueeze(1)
    if len(mask_total.shape) == 2:
        mask_total = mask_total.squeeze(1)
    else:
        print("2:1 ", mask_total.shape)
    correct = torch.sum((indices[mask_total] == logits_labels[mask_total]))
    return correct.item() * 1.0 / len(logits_labels[mask_total])
