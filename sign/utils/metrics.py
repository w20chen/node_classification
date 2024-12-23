import torch
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, normalized_mutual_info_score, adjusted_rand_score

__all__ = ['accuracy', 'micro_macro_f1_score', 'nmi_ari_score', 'mean_reciprocal_rank', 'hits_at']


def accuracy(logits, labels):
    """Calculate accuracy

    :param logits: tensor(N, C) predicted probabilities, N is the number of samples, C is the number of classes
    :param labels: tensor(N) true labels
    :return: float accuracy
    """
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)


def micro_macro_f1_score(logits, labels):
    """Calculate Micro-F1 and Macro-F1 scores

    :param logits: tensor(N, C) predicted probabilities, N is the number of samples, C is the number of classes
    :param labels: tensor(N) true labels
    :return: float, float Micro-F1 and Macro-F1 scores
    """
    prediction = torch.argmax(logits, dim=1).long().numpy()
    labels = labels.numpy()
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')
    return micro_f1, macro_f1


def nmi_ari_score(logits, labels):
    """Calculate NMI and ARI scores

    :param logits: tensor(N, d) predicted probabilities, N is the number of samples
    :param labels: tensor(N) true labels
    :return: float, float NMI and ARI scores
    """
    num_classes = logits.shape[1]
    prediction = KMeans(n_clusters=num_classes).fit_predict(logits.detach().numpy())
    labels = labels.numpy()
    nmi = normalized_mutual_info_score(labels, prediction)
    ari = adjusted_rand_score(labels, prediction)
    return nmi, ari


def mean_reciprocal_rank(predicts, answers):
    """Calculate Mean Reciprocal Rank (MRR) = sum(1 / rank_i) / N, if the answer is not in the predictions, the rank is considered as ∞

    For example, predicts=[[2, 0, 1], [2, 1, 0], [1, 0, 2]], answers=[1, 2, 8],
    ranks=[3, 2, ∞], MRR=(1/3+1/1+0)/3=4/9

    :param predicts: tensor(N, K) predicted results, N is the number of samples, K is the number of predictions
    :param answers: tensor(N) positions of the correct answers
    :return: float MRR∈[0, 1]
    """
    ranks = torch.nonzero(predicts == answers.unsqueeze(1), as_tuple=True)[1] + 1
    return torch.sum(1.0 / ranks.float()).item() / len(predicts)


def hits_at(n, predicts, answers):
    """Calculate Hits@n = #(rank_i <= n) / N

    For example, predicts=[[2, 0, 1], [2, 1, 0], [1, 0, 2]], answers=[1, 2, 0], ranks=[3, 1, 2], Hits@2=2/3

    :param n: int the proportion of answers ranked within the top n
    :param predicts: tensor(N, K) predicted results, N is the number of samples, K is the number of predictions
    :param answers: tensor(N) positions of the correct answers
    :return: float Hits@n∈[0, 1]
    """
    ranks = torch.nonzero(predicts == answers.unsqueeze(1), as_tuple=True)[1] + 1
    return torch.sum(ranks <= n).float().item() / len(predicts)
