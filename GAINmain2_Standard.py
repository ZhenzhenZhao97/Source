# %% md

# Creator: Dhanajit
# Brahma
#
# Adapted
# from the original
#
# implementation in tensorflow
# from here: https: // github.com / jsyoon0823 / GAIN
#
# Generative
# Adversarial
# Imputation
# Networks(GAIN)
# Implementation
# on
# Letter and Spam
# Dataset
#
# Reference: J.Yoon, J.Jordon, M.van
# der
# Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets,"
# ICML, 2018.

# %%

# %% Packages
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %%
from sklearn.preprocessing import MinMaxScaler, StandardScaler

torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True

dataset_file = 'Speed_325.csv'  # 'Letter.csv' for Letter dataset an 'Spam.csv' for Spam dataset
use_gpu = False  # set it to True to use GPU and False to use CPU
if use_gpu:
    torch.cuda.set_device(0)

# %%

# %% System Parameters
# 1. Mini batch size
mb_size = 128
# 2. Missing rate
p_miss = 0.2
# 3. Hint rate
p_hint = 0.9
# 4. Loss Hyperparameters
alpha = 10
# 5. Train Rate
train_rate = 0.8

# %% Data

# Data generation
Data = np.loadtxt(dataset_file, delimiter=",", skiprows=0)

# Parameters
No = len(Data)
Dim = len(Data[0, :])

# Hidden state dimensions
H_Dim1 = Dim
H_Dim2 = Dim

# Normalization (0 to 1)
Min_Val = np.zeros(Dim)
Max_Val = np.zeros(Dim)
# sclar = MinMaxScaler()
sclar = StandardScaler()
Data = sclar.fit_transform(Data)
# for i in range(Dim):
#     Min_Val[i] = np.min(Data[:, i])
#     Data[:, i] = Data[:, i] - np.min(Data[:, i])
#     Max_Val[i] = np.max(Data[:, i])
#     Data[:, i] = Data[:, i] / (np.max(Data[:, i]) + 1e-6)

# %% Missing introducing
p_miss_vec = p_miss * np.ones((Dim, 1))

Missing = np.zeros((No, Dim))

for i in range(Dim):
    A = np.random.uniform(0., 1., size=[len(Data), ])
    B = A > p_miss_vec[i]
    Missing[:, i] = 1. * B

# %% Train Test Division

idx = np.random.permutation(No)

Train_No = int(No * train_rate)
Test_No = No - Train_No

# Train / Test Features
trainX = Data[idx[:Train_No], :]
testX = Data[idx[Train_No:], :]

# Train / Test Missing Indicators
trainM = Missing[idx[:Train_No], :]
testM = Missing[idx[Train_No:], :]


# %% Necessary Functions

# 1. Xavier Initialization Definition
# def xavier_init(size):
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape = size, stddev = xavier_stddev)
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return np.random.normal(size=size, scale=xavier_stddev)


# Hint Vector Generation
def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C

def my_loss(X, M, New_X):
    #%% Structure
    # Generator
    G_sample = generator(New_X, M)
    # Renormalization
    Max_Val_torch = torch.from_numpy(Max_Val+ 1e-6) #.to(device=torch.device("cuda"))
    Min_Val_torch = torch.from_numpy(Min_Val) # .to(device=torch.device("cuda"))
    X = X * (Max_Val_torch) + Min_Val_torch
    G_sample = G_sample * (Max_Val_torch) + Min_Val_torch
    ori_data = X.cpu().numpy()
    imputed_data = G_sample.detach().cpu().numpy()
    data_m = M.cpu().numpy()

    index = np.where((ori_data != 0) & (data_m == 0))

    denominator = np.sum(1 - data_m[index])

    nominator = np.sum(np.abs((1 - data_m[index]) * ori_data[index] - (1 - data_m[index]) * imputed_data[index]))

    nominator_2 = np.sum(((1 - data_m[index]) * ori_data[index] - (1 - data_m[index]) * imputed_data[index]) ** 2)

    nominator_3 = np.sum(np.abs((1 - data_m[index]) * ori_data[index] - (1 - data_m[index]) * imputed_data[index]) / (
            (1 - data_m[index]) * ori_data[index]))

    RMSE_test_loss = np.sqrt(nominator_2 / float(denominator))

    MAPE_test_loss = nominator_3/denominator

    MAE_test_loss = nominator / denominator
    #%% MSE Performance metric
    # MSE_test_loss = torch.mean(((1-M) * X - (1-M)*G_sample)**2) / torch.mean(1-M)
    # MAE_test_loss = torch.mean(torch.abs((1 - M) * X - (1 - M) * G_sample)) / torch.mean(1 - M)
    # MAPE_test_loss = torch.mean(torch.abs(((1 - M) * X - (1 - M) * G_sample))/(1 - M) * X) / torch.mean(1 - M)
    return RMSE_test_loss, MAE_test_loss, MAPE_test_loss, G_sample

# %% md

### GAIN Architecture
# GAIN
# Consists
# of
# 3
# Components
# - Generator
# - Discriminator
# - Hint
# Mechanism

# %%

# %% 1. Discriminator
if use_gpu is True:
    D_W1 = torch.tensor(xavier_init([Dim * 2, H_Dim1]), requires_grad=True, device="cuda")  # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True, device="cuda")

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True, device="cuda")
    D_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True, device="cuda")

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True, device="cuda")
    D_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True, device="cuda")  # Output is multi-variate
else:
    D_W1 = torch.tensor(xavier_init([Dim * 2, H_Dim1]), requires_grad=True)  # Data + Hint as inputs
    D_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True)

    D_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True)
    D_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True)

    D_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True)
    D_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True)  # Output is multi-variate

theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

# %% 2. Generator
if use_gpu is True:
    G_W1 = torch.tensor(xavier_init([Dim * 2, H_Dim1]), requires_grad=True,
                        device="cuda")  # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True, device="cuda")

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True, device="cuda")
    G_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True, device="cuda")

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True, device="cuda")
    G_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True, device="cuda")
else:
    G_W1 = torch.tensor(xavier_init([Dim * 2, H_Dim1]),
                        requires_grad=True)  # Data + Mask as inputs (Random Noises are in Missing Components)
    G_b1 = torch.tensor(np.zeros(shape=[H_Dim1]), requires_grad=True)

    G_W2 = torch.tensor(xavier_init([H_Dim1, H_Dim2]), requires_grad=True)
    G_b2 = torch.tensor(np.zeros(shape=[H_Dim2]), requires_grad=True)

    G_W3 = torch.tensor(xavier_init([H_Dim2, Dim]), requires_grad=True)
    G_b3 = torch.tensor(np.zeros(shape=[Dim]), requires_grad=True)

theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]


# %% md

## GAIN Functions

# %%

# %% 1. Generator
def generator(new_x, m):
    inputs = torch.cat(dim=1, tensors=[new_x, m])  # Mask + Data Concatenate
    G_h1 = F.relu(torch.matmul(inputs, G_W1) + G_b1)
    G_h2 = F.relu(torch.matmul(G_h1, G_W2) + G_b2)
    G_prob = torch.sigmoid(torch.matmul(G_h2, G_W3) + G_b3)  # [0,1] normalized Output

    return G_prob


# %% 2. Discriminator
def discriminator(new_x, h):
    inputs = torch.cat(dim=1, tensors=[new_x, h])  # Hint + Data Concatenate
    D_h1 = F.relu(torch.matmul(inputs, D_W1) + D_b1)
    D_h2 = F.relu(torch.matmul(D_h1, D_W2) + D_b2)
    D_logit = torch.matmul(D_h2, D_W3) + D_b3
    D_prob = torch.sigmoid(D_logit)  # [0,1] Probability Output

    return D_prob


# %% 3. Other functions
# Random sample generator for Z
def sample_Z(m, n):
    return np.random.uniform(0., 0.01, size=[m, n])


# Mini-batch generation
def sample_idx(m, n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx


# %% md

## GAIN Losses

# %%

def discriminator_loss(M, New_X, H):
    # Generator
    G_sample = generator(New_X, M)
    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    # %% Loss
    D_loss = -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))
    return D_loss


def generator_loss(X, M, New_X, H):
    # %% Structure
    # Generator
    G_sample = generator(New_X, M)

    # Combine with original data
    Hat_New_X = New_X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_New_X, H)

    # %% Loss
    G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((M * New_X - M * G_sample) ** 2) / torch.mean(M)

    G_loss = G_loss1 + alpha * MSE_train_loss

    # %% MSE Performance metric
    MSE_test_loss = torch.mean(((1 - M) * X - (1 - M) * G_sample) ** 2) / torch.mean(1 - M)
    return G_loss, MSE_train_loss, MSE_test_loss



# %% md

## Optimizers

# %%

optimizer_D = torch.optim.Adam(params=theta_D)
optimizer_G = torch.optim.Adam(params=theta_G)

# %% md

## Training

# %%

# %% Start Iterations
for it in tqdm(range(5001)):

    # %% Inputs
    mb_idx = sample_idx(Train_No, mb_size)
    X_mb = trainX[mb_idx, :]

    Z_mb = sample_Z(mb_size, Dim)
    M_mb = trainM[mb_idx, :]
    H_mb1 = sample_M(mb_size, Dim, 1 - p_hint)
    H_mb = M_mb * H_mb1

    New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

    if use_gpu is True:
        X_mb = torch.tensor(X_mb, device="cuda")
        M_mb = torch.tensor(M_mb, device="cuda")
        H_mb = torch.tensor(H_mb, device="cuda")
        New_X_mb = torch.tensor(New_X_mb, device="cuda")
    else:
        X_mb = torch.tensor(X_mb)
        M_mb = torch.tensor(M_mb)
        H_mb = torch.tensor(H_mb)
        New_X_mb = torch.tensor(New_X_mb)
    # print(X_mb)
    optimizer_D.zero_grad()
    D_loss_curr = discriminator_loss(M=M_mb, New_X=New_X_mb, H=H_mb)
    D_loss_curr.backward()
    optimizer_D.step()

    optimizer_G.zero_grad()
    G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(X=X_mb, M=M_mb, New_X=New_X_mb, H=H_mb)
    G_loss_curr.backward()
    optimizer_G.step()

    # %% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
        print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
        print(G_loss_curr.item())
        print(D_loss_curr.item())
        print()

# %% md

Z_mb = sample_Z(Test_No, Dim)
M_mb = testM
X_mb = testX

New_X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb  # Missing Data Introduce

if use_gpu is True:
    X_mb = torch.tensor(X_mb, device='cuda')
    M_mb = torch.tensor(M_mb, device='cuda')
    New_X_mb = torch.tensor(New_X_mb, device='cuda')
else:
    X_mb = torch.tensor(X_mb)
    M_mb = torch.tensor(M_mb)
    New_X_mb = torch.tensor(New_X_mb)

RMSE_final, MAE_final, MAPE_final, Sample = my_loss(X=X_mb, M=M_mb, New_X=New_X_mb)

print('Final Test RMSE: ' + str(RMSE_final.item()))
print('Final Test MAE: ' + str(MAE_final.item()))
print('Final Test MAPE: ' + str(MAPE_final.item()))
# %%

imputed_data = M_mb * X_mb + (1 - M_mb) * Sample
print("Imputed test data:")
# np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

if use_gpu is True:
    print(imputed_data.cpu().detach().numpy())
else:
    print(imputed_data.detach().numpy())

'''
0.2
Final Test RMSE: 3.73130574591619
Final Test MAE: 2.10871885261881
Final Test MAPE: 0.04502733845209246

0.4
Final Test RMSE: 4.014233475994281
Final Test MAE: 2.269088960841889
Final Test MAPE: 0.04816150578193554

0.6
Final Test RMSE: 14.326643230509378
Final Test MAE: 8.12028928958589
Final Test MAPE: 0.14002699534187238

0.8
--
Final Test RMSE: 27.55279458901586
Final Test MAE: 21.02456614477012
Final Test MAPE: 0.3432329274953421
'''