import numpy as np


def create_reward_penalty_graph(ph_dict, labels, train_idx, val_idx, test_idx, args):
    scores = args.scores
    num_nodes = len(labels)

    # reward graph: subjects i and j have the same label and score, r(i,j) + 1
    # penalty_graph: subjects i and j have different labels but the same scores, p(i,j) + 1
    # motivation_graph: the labels of subjects i and j are unknown and their scores are the same, m(i,j) + 1
    reward_graph = np.zeros((len(scores), num_nodes, num_nodes))
    penalty_graph = np.zeros((len(scores), num_nodes, num_nodes))
    motivation_graph = np.zeros((len(scores), num_nodes, num_nodes))

    for l, score in enumerate(scores):
        label_dict = ph_dict[score]
        if score in [args.ages, 'FIQ']:
            for i in train_idx:
                for j in train_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[j]))
                        if val < 2 and (labels[i] == labels[j]):
                            reward_graph[l, i, j] += 1
                        elif val < 2 and (labels[i] != labels[j]):
                            penalty_graph[l, i, j] += 1
                    except ValueError:  # missing label
                        pass
                for k in val_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[k]))
                        if val < 2:
                            motivation_graph[l, i, k] += 1
                            motivation_graph[l, k, i] += 1
                    except ValueError:  # missing label
                        pass
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                            motivation_graph[l, v, i] += 1
                    except ValueError:  # missing label
                        pass

            for i in val_idx:
                for k in val_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[k]))
                        if val < 2:
                            motivation_graph[l, i, k] += 1
                    except ValueError:  # missing label
                        pass
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                            motivation_graph[l, v, i] += 1
                    except ValueError:  # missing label
                        pass

            for i in test_idx:
                for v in test_idx:
                    try:
                        val = abs(float(label_dict[i]) - float(label_dict[v]))
                        if val < 2:
                            motivation_graph[l, i, v] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for i in train_idx:
                for j in train_idx:
                    if (label_dict[i] == label_dict[j]) and (labels[i] == labels[j]):
                        reward_graph[l, i, j] += 1
                    elif (label_dict[i] == label_dict[j]) and (labels[i] != labels[j]):
                        penalty_graph[l, i, j] += 1
                for k in val_idx:
                    motivation_graph[l, i, k] += 1
                    motivation_graph[l, k, i] += 1
                for v in test_idx:
                    motivation_graph[l, i, v] += 1
                    motivation_graph[l, v, i] += 1

            for i in val_idx:
                for k in val_idx:
                    motivation_graph[l, i, k] += 1
                for v in test_idx:
                    motivation_graph[l, i, v] += 1
                    motivation_graph[l, v, i] += 1

            for i in test_idx:
                for v in test_idx:
                    motivation_graph[l, i, v] += 1

        final_graph = np.stack((reward_graph, penalty_graph, motivation_graph), axis=1)
        return final_graph
