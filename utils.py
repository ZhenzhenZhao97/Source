import torch
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler, MinMaxScaler

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
np.random.seed(2333)


def load_raw_data(file_path):
    Data = np.loadtxt(file_path, delimiter=",", skiprows=0)
    Dim = len(Data[0, :])
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(Data[:, i])
        Data[:, i] = Data[:, i] - np.min(Data[:, i])
        Max_Val[i] = np.max(Data[:, i])
        Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)
    return Data


def load_raw_data_2(file_path):
    Data = np.loadtxt(file_path, delimiter=",", skiprows=0)
    #print(Data)
    Dim = len(Data[0, :])
    Min_Val = np.zeros(Dim)
    Max_Val = np.zeros(Dim)

    for i in range(Dim):
        Min_Val[i] = np.min(Data[:, i])
        Data[:, i] = Data[:, i] - np.min(Data[:, i])
        Max_Val[i] = np.max(Data[:, i])
        Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)
    return Data, Min_Val, Max_Val


def get_adjacency_matrix(distance_df_filename, num_of_vertices):
    with open(distance_df_filename, 'r') as f:
        reader = csv.reader(f)
        header = f.__next__()
        edges = [(int(i[0]), int(i[1]), float(i[2])) for i in reader]

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    for i, j, z in edges:
        # z = z / 100  # 将权重映射到（0，1）用于下文判别
        # z2 = z * z
        # A[i, j] = np.exp(-z2 / 10)
        # A[j, i] = np.exp(-z2 / 10)
        A[i, j] = 1
        A[j, i] = 1

    return A


def get_adj_bay_matrix(distance_df_filename, num_of_vertices):
    A = np.load(distance_df_filename)
    A = A - np.identity(num_of_vertices)
    return A


def weight_matrix(file_path, sigma2=0.1, epsilon=0.5, scaling=True):
    try:
        W = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    # check whether W is a 0/1 matrix.
    if set(np.unique(W)) == {0, 1}:
        print('The input graph is a 0/1 matrix; set "scaling" to False.')
        scaling = False

    if scaling:
        n = W.shape[0]
        W = W / 10000.
        W2, W_mask = W * W, np.ones([n, n]) - np.identity(n)
        # refer to Eq.10
        return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask
    else:
        return W


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam-np.eye(n)


def get_mask_matrix(rate, arr):  # arr_size 表示数组中所有元素的个数
    zeros_nums = int(arr.size * rate)
    new_matrix = np.ones(arr.size)
    new_matrix[: zeros_nums] = 0
    np.random.shuffle(new_matrix)
    return new_matrix.reshape(arr.shape)


def get_data_matrix(raw_data_matrix, rate):
    np_data = np.array(raw_data_matrix)
    mask_matrix = get_mask_matrix(rate=rate, arr=np_data)
    data_matrix = np.multiply(raw_data_matrix, mask_matrix)
    return data_matrix, np.array(mask_matrix)


def load_data(file_path, len_train, len_val, rate):
    df = load_raw_data(file_path)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]

    train_data, train_m = get_data_matrix(train, rate)
    val_data, val_m = get_data_matrix(val, rate)
    test_data, test_m = get_data_matrix(test, rate)

    return train, val, test, train_data, val_data, test_data, train_m, val_m, test_m


def data_transform_label(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def data_transform_train(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def mask_transform_min_batch(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def random_transform_min_batch(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    n_slot = day_slot - n_his - n_pred + 1
    x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    for i in range(n_day):
        for j in range(n_slot):
            t = i * n_slot + j
            s = i * day_slot + j
            e = s + n_his
            x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    return torch.Tensor(x).to(device)


def mse_test_loss(X, M, Label):
    MSE_test_loss = torch.mean(((1-M) * X - (1-M)*Label)**2) / torch.mean(1-M)
    return MSE_test_loss


def metric_test(X, M, Label):
    MAE_test_loss = torch.mean(torch.abs((1 - M) * X - (1 - M) * Label)) / torch.mean(1 - M)
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * Label) ** 2) / torch.mean(1 - M)
    index = torch.where((Label != 0) & (M == 0))
    MAPE_test_loss = torch.sum(torch.abs(Label[index]-X[index])/Label[index])/Label[index].shape[0]
    return MSE_test_loss, MAPE_test_loss * 100, MAE_test_loss


def mse_train_loss(X, M, Label):
    MSE_train_loss = torch.mean((M * X - M * Label) ** 2) / torch.mean(M)
    return MSE_train_loss


def evaluate_model(model, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            label_y = y.squeeze()
            mask = x[:, 1, :, :]
            y_pred = model(x[:, 0, :, :].unsqueeze(1))
            l_loss = mse_test_loss(y_pred, mask, label_y)
            l_sum += l_loss.item()* y.shape[0]
            n +=  y.shape[0]
        return l_sum / n


def evaluate_model_2(model, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            label_y = y[:, 0, :, :]
            mask = y[:, 1, :, :]
            y_pred = model(x)
            l_loss = mse_test_loss(X=y_pred, M=mask, Label=label_y)
            l_sum += l_loss.item()* y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter,file_path):
    model.eval()
    _, MinValue, MaxValue = load_raw_data_2(file_path)
    MinValue = torch.tensor(MinValue).to(device)
    MaxValue = torch.tensor(MaxValue).to(device)
    with torch.no_grad():
        l_sum, mape_sum, mae_sum, n = 0.0, 0.0, 0.0, 0
        # ground_truth = []
        # y_pred_value = []
        # mask_value = []
        for x, y in data_iter:
            label_y = y[:, 0, :, :]
            mask = y[:, 1, :, :]
            y_pred = model(x)
            y_pred = y_pred * (MaxValue + 1e-6) + MinValue
            label_y = label_y * (MaxValue + 1e-6) + MinValue
            # ground_truth.extend(label_y[:, -1, 300].tolist())
            # y_pred_value.extend(y_pred[:, -1, 300].tolist())
            # mask_value.extend(mask[:, -1, 300].tolist())
            # l_loss = mse_test_loss(y_pred, mask, label_y)
            mse, mape, mae = metric_test(X=y_pred, M=mask, Label=label_y)
            l_sum += mse.item() * y.shape[0]
            mape_sum += mape.item() * y.shape[0]
            mae_sum += mae.item() * y.shape[0]
            #l_sum += l_loss.item() * y.shape[0]
            n += y.shape[0]
        # pd_y_pred_value = pd.DataFrame(y_pred_value)
        # pd_ground_truth = pd.DataFrame(ground_truth)
        # pd_mask_value = pd.DataFrame(mask_value)
        # pd_y_pred_value.to_csv("./pretrain_model/pemsbay/prediction.csv",header=False,index=False)
        # pd_ground_truth.to_csv("./pretrain_model/pemsbay/ground_truth.csv",header=False,index=False)
        # pd_mask_value.to_csv("./pretrain_model/pemsbay/mask_value.csv", header=False, index=False)
        RMSE = np.sqrt(np.array(l_sum / n).mean())
        MAPE = np.array(mape_sum/n).mean()
        MAE = np.array(mae_sum/n).mean()
        return RMSE, MAPE, MAE


def evaluate_metric_2(model, data_iter, file_path):
    model.eval()
    _, MinValue, MaxValue = load_raw_data_2(file_path)
    MinValue = torch.tensor(MinValue).to(device)
    MaxValue = torch.tensor(MaxValue).to(device)
    with torch.no_grad():
        l_sum, mape_sum, mae_sum, n = 0.0, 0.0, 0.0, 0
        ground_truth = [[] for i in range(325)]
        y_pred_value = [[] for i in range(325)]
        mask_value = [[] for i in range(325)]
        for x, y in data_iter:
            label_y = y[:, 0, :, :]
            mask = y[:, 1, :, :]
            y_pred = model(x)
            y_pred = y_pred * (MaxValue + 1e-6) + MinValue
            label_y = label_y * (MaxValue + 1e-6) + MinValue
            #print(label_y)
            for i in range(325):
                ground_truth[i].extend(label_y[:, -1, i].tolist())
                y_pred_value[i].extend(y_pred[:, -1, i].tolist())
                mask_value[i].extend(mask[:, -1, i].tolist())
            # l_loss = mse_test_loss(y_pred, mask, label_y)
            mse, mape, mae = metric_test(X=y_pred, M=mask, Label=label_y)
            l_sum += mse.item() * y.shape[0]
            mape_sum += mape.item() * y.shape[0]
            mae_sum += mae.item() * y.shape[0]
            #l_sum += l_loss.item() * y.shape[0]
            n += y.shape[0]
        pd_y_pred_value = pd.DataFrame(y_pred_value)
        pd_ground_truth = pd.DataFrame(ground_truth)
        pd_mask_value = pd.DataFrame(mask_value)
        pd_y_pred_value.to_csv("./pretrain_model/pemsbay/prediction.csv",header=False,index=False)
        pd_ground_truth.to_csv("./pretrain_model/pemsbay/ground_truth.csv",header=False,index=False)
        pd_mask_value.to_csv("./pretrain_model/pemsbay/mask_value.csv", header=False, index=False)
        RMSE = np.sqrt(np.array(l_sum / n).mean())
        MAPE = np.array(mape_sum/n).mean()
        MAE = np.array(mae_sum/n).mean()
        return RMSE, MAPE, MAE


def evaluate_metric_3(model, data_iter, file_path):
    model.eval()
    _, MinValue, MaxValue = load_raw_data_2(file_path)
    MinValue = torch.tensor(MinValue).to(device)
    MaxValue = torch.tensor(MaxValue).to(device)
    with torch.no_grad():
        l_sum, mape_sum, mae_sum, n = 0.0, 0.0, 0.0, 0
        ground_truth = [[] for i in range(307)]
        y_pred_value = [[] for i in range(307)]
        mask_value = [[] for i in range(307)]
        for x, y in data_iter:
            label_y = y[:, 0, :, :]
            mask = y[:, 1, :, :]
            y_pred = model(x)
            y_pred = y_pred * (MaxValue + 1e-6) + MinValue
            label_y = label_y * (MaxValue + 1e-6) + MinValue

            for i in range(307):
                ground_truth[i].extend(label_y[:, -1, i].tolist())
                y_pred_value[i].extend(y_pred[:, -1, i].tolist())
                mask_value[i].extend(mask[:, -1, i].tolist())
            # l_loss = mse_test_loss(y_pred, mask, label_y)
            mse, mape, mae = metric_test(X=y_pred, M=mask, Label=label_y)
            l_sum += mse.item() * y.shape[0]
            mape_sum += mape.item() * y.shape[0]
            mae_sum += mae.item() * y.shape[0]
            #l_sum += l_loss.item() * y.shape[0]
            n += y.shape[0]
        pd_y_pred_value = pd.DataFrame(y_pred_value)
        pd_ground_truth = pd.DataFrame(ground_truth)
        pd_mask_value = pd.DataFrame(mask_value)
        pd_y_pred_value.to_csv("./pretrain_model/pemsd4/prediction.csv",header=False,index=False)
        pd_ground_truth.to_csv("./pretrain_model/pemsd4/ground_truth.csv",header=False,index=False)
        pd_mask_value.to_csv("./pretrain_model/pemsd4/mask_value.csv", header=False, index=False)
        RMSE = np.sqrt(np.array(l_sum / n).mean())
        MAPE = np.array(mape_sum/n).mean()
        MAE = np.array(mae_sum/n).mean()
        return RMSE, MAPE, MAE