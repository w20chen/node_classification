import sys
import matplotlib.pyplot as plt
import random
from CoraData import *
from MyGCNNet import *
from MyGATNet import *
from CiteseerData import CiteseerData
from ogbData import *


# Generate tensor_adjacency
def generate_tensor_adjacency_for_classify(edge_index, drop_edge=1.1):
    if drop_edge >= 1.0:
        adj = get_adjacent(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True)
    else:
        adj = random_adjacent_sampler(edge_of_pg=edge_index, num_graph_node=num_nodes, symmetric_of_edge=True,
                                      drop_edge=drop_edge)

    normalize_adj = normalization(adj, self_link=True)

    # Prepare to convert the original coo_matrix to tensor form
    index_of_coo_matrix = torch.from_numpy(np.asarray([normalize_adj.row,
                                                       normalize_adj.col]).astype('int64')).long()

    values_of_index_in_matrix = torch.from_numpy(normalize_adj.data.astype(np.float32))

    # Construct a sparse matrix tensor, tensor size is (2708,2708)
    tensor_adjacency = torch.sparse_coo_tensor(
        index_of_coo_matrix, values_of_index_in_matrix,
        torch.Size([num_nodes, num_nodes]))
    return tensor_adjacency


# Set random seed
def init_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def plot_loss_with_acc(loss_history, val_acc_history):
    fig = plt.figure()
    # ax1 draws curve 1
    ax1 = fig.add_subplot(111)  # Indicates dividing the plot interface into 1 row and 1 column, this subplot occupies position 1 from left to right, top to bottom
    ax1.plot(range(len(loss_history)), loss_history,
             c=np.array([255, 71, 90]) / 255.)  # c is color
    plt.ylabel('Loss')

    # ax2 draws curve 2
    ax2 = fig.add_subplot(111, sharex=ax1, frameon=False)  # Essentially adding a coordinate system, setting to share ax1's x-axis, ax2 background transparent
    ax2.plot(range(len(val_acc_history)), val_acc_history,
             c=np.array([79, 179, 255]) / 255.)
    ax2.yaxis.tick_right()  # Enable right y-axis

    ax2.yaxis.set_label_position("right")
    plt.ylabel('ValAcc')

    plt.xlabel('Epoch')
    plt.title('Training Loss & Validation Accuracy')
    plt.show()


if __name__ == '__main__':
    args = sys.argv

    def train():
        loss_list = []
        val_acc_history = []
        model.train()

        train_y = tensor_y[train_mask]
        tensor_adjacency = generate_tensor_adjacency_for_classify(edge_index=edge_index, drop_edge=drop_edge).to(device)

        for epoch in range(epoch_num):
            logits = model(tensor_x, tensor_adjacency)
            train_mask_logits = logits[train_mask]

            loss = criterion(train_mask_logits, train_y.long())     # Calculate loss
            optimizer.zero_grad()                                   # Zero gradients
            loss.backward()                                         # Backpropagation to compute parameter gradients
            optimizer.step()                                        # Use optimization method for gradient update

            train_acc = test(train_mask)                            # Calculate current model accuracy on training set, call test function
            val_acc = test(val_mask)                                # Calculate current model accuracy on validation set

            # Record changes in loss and accuracy during training for plotting
            loss_list.append(loss.item())
            val_acc_history.append(val_acc.item())
            print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
                epoch, loss.item(), train_acc.item(), val_acc.item()))

        return loss_list, val_acc_history


    def test(mask):
        model.eval()  # Indicate switching model to evaluation (test) mode, thus excluding BN and Dropout interference during testing

        tensor_adjacency = generate_tensor_adjacency_for_classify(edge_index=edge_index).to(device)

        with torch.no_grad():  # Significantly reduce memory usage
            logits = model(tensor_x, tensor_adjacency)
            test_mask_logits = logits[mask]

            predict_y = test_mask_logits.max(1)[1]  # Return index of maximum value in each row (return column index of maximum element in each row)
            accuracy = torch.eq(predict_y, tensor_y[mask]).float().mean()
        
        return accuracy


    init_seeds()
    # Set platform to run on CPU or GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Hyperparameter settings
    learning_rate = 0.001
    epoch_num = 100
    weight_decay = 5e-4
    hidden_layer_dim = 512
    layer_num = 2
    drop_edge = 0.05
    use_pair_norm = True

    dataset_name = "cora"
    if len(args) == 2:
        dataset_name = args[1]

    path_of_cora = "./cora/cora"
    path_of_citeseer = "./citeseer/citeseer"
    path_of_ogb = "./ogbn-products/products/raw"

    print(f'Reading dataset {dataset_name}...')

    dataset = None
    if dataset_name == "cora":
        dataset = CoraData(path_of_cora)
    elif dataset_name == "citeseer":
        dataset = CiteseerData(path_of_citeseer)
    elif dataset_name == "ogb":
        dataset = ogbData(path_of_ogb)
    else:
        print(f'Invalid dataset name {dataset}')
        exit()

    num_nodes = dataset.num_nodes
    edge_index = dataset.edge_of_pg
    train_mask, val_mask, test_mask = dataset.data_partition_node()
    num_of_class = dataset.num_of_class
    feature_dim = dataset.feature_dim

    tensor_x = torch.tensor(dataset.feature_of_pg, device=device, dtype=torch.float)    # shape: (num_nodes, feature_dim)
    tensor_y = torch.tensor(dataset.label_of_pg, device=device, dtype=torch.float)      # shape: (num_nodes,)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)
    print(f'train {train_mask.shape}, validate {val_mask.shape}, test {test_mask.shape}')

    model = ClassificationSAGEFromPYG(
        hidden_layer_dim=hidden_layer_dim,
        num_of_hidden_layer=layer_num,
        use_pair_norm=use_pair_norm,
        num_of_class=num_of_class,
        input_feature_dim=feature_dim
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss, val_acc = train()
    test_acc = test(test_mask)
    print("Test accuracy: ", test_acc.item())
    plot_loss_with_acc(loss, val_acc)
