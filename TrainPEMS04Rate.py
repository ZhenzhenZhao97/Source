import random
import torch.nn as nn
from utils import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from auto_encoder import AutoEncoderModel
from tcn import TemporalConvNet


torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "./datasets/pemsd4/Speed_307.csv"
weight_path = "./datasets/pemsd4/distance.csv"
save_path = "./pretrain_model/pemsd4/model_pemsd4_04_3.pt"
n_train, n_val, n_test = 49, 5, 5
day_slot = 288
node_nums = 307
missing_rate = 0.4
n_his = 12
n_pred = 0
batch_size = 64
epochs = 50
kernel_size = 3
embedding_size = 32
num_channels = [16, 32, 16]
input_feature = 1
lr = 1e-3
Ks = 3
W = get_adjacency_matrix(weight_path, node_nums)
# print(W)
L = scaled_laplacian(W)
Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
train, val, test, train_data, val_data, test_data, train_m, val_m, test_m = load_data(file_path=data_path,
                                                                                      len_train=n_train * day_slot,
                                                                                      len_val=n_val * day_slot,
                                                                                      rate=missing_rate)


y_train = data_transform_label(train, n_his, n_pred, day_slot, device)
y_val = data_transform_label(val, n_his, n_pred, day_slot, device)
y_test = data_transform_label(test, n_his, n_pred, day_slot, device)

x_train = data_transform_train(train_data, n_his, n_pred, day_slot, device)
x_val = data_transform_train(val_data, n_his, n_pred, day_slot, device)
x_test = data_transform_train(test_data, n_his, n_pred, day_slot, device)

train_mask = mask_transform_min_batch(train_m, n_his, n_pred, day_slot, device)
val_mask = mask_transform_min_batch(val_m, n_his, n_pred, day_slot, device)
test_mask = mask_transform_min_batch(test_m, n_his, n_pred, day_slot, device)


y_train_concat = torch.cat([y_train, train_mask], dim=1)   # 将掩盖矩阵包含进去，以便后续更新函数
y_val_concat = torch.cat([y_val, val_mask], dim=1)
y_test_concat = torch.cat([y_test, test_mask], dim=1)



train_data = TensorDataset(x_train, y_train_concat)
train_iter = DataLoader(train_data, batch_size, shuffle=True)
val_data = TensorDataset(x_val, y_val_concat)
val_iter = DataLoader(val_data, batch_size)
test_data = TensorDataset(x_test, y_test_concat)
test_iter = DataLoader(test_data, batch_size)

model = AutoEncoderModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
                         num_channels=num_channels, Lk=Lk, kernel_size=kernel_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
min_val_loss = np.inf

for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        label_y = y[:, 0, :, :]
        mask = y[:, 1, :, :]
        # print(x)
        # print(label_y)
        model_outputs = model(x)
        l_loss = mse_train_loss(model_outputs, mask, label_y)
        optimizer.zero_grad()
        l_loss.backward()
        optimizer.step()
        l_sum += l_loss.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model_2(model, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

# best_model = AutoEncoderModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
#                          num_channels=num_channels, Lk=Lk, kernel_size=kernel_size).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model_2(best_model, test_iter)
# print("Test_loss:", l)
# RMSE, MAPE, MAE = evaluate_metric(best_model, test_iter,data_path)
# print("RMSE:", RMSE, "MAPE:", MAPE, "MAE:", MAE)

'''epoch 1 , train loss: 0.09140752528239077 , validation loss: 0.013921073267391011
epoch 2 , train loss: 0.008973666385908808 , validation loss: 0.009307300345467853
epoch 3 , train loss: 0.006398650879601775 , validation loss: 0.007280519531387499
epoch 4 , train loss: 0.004927941438432044 , validation loss: 0.005300495832055699
epoch 5 , train loss: 0.003694791381267858 , validation loss: 0.005317803677877525
epoch 6 , train loss: 0.0031011439474932325 , validation loss: 0.004157610008634277
epoch 7 , train loss: 0.0027393652866900863 , validation loss: 0.003833811175219365
epoch 8 , train loss: 0.002443026623259858 , validation loss: 0.0034575977268551444
epoch 9 , train loss: 0.0021675270714999564 , validation loss: 0.0033653108528327695
epoch 10 , train loss: 0.001962787797628221 , validation loss: 0.0031988319401235893
epoch 11 , train loss: 0.0017849351410665305 , validation loss: 0.0035099347981049373
epoch 12 , train loss: 0.0017068913676956796 , validation loss: 0.00307036792205492
epoch 13 , train loss: 0.001646555856354057 , validation loss: 0.0030484476966433738
epoch 14 , train loss: 0.0015913889574798621 , validation loss: 0.0028920486619837234
epoch 15 , train loss: 0.0015600780856662563 , validation loss: 0.0028769941075910085
epoch 16 , train loss: 0.0015104434187390562 , validation loss: 0.002855132162547418
epoch 17 , train loss: 0.001487505344159647 , validation loss: 0.002875046753190081
epoch 18 , train loss: 0.0014702164967385393 , validation loss: 0.002949852271960369
epoch 19 , train loss: 0.0014592689126611946 , validation loss: 0.0028422618294067495
epoch 20 , train loss: 0.001439329563515829 , validation loss: 0.0028779451656306587
epoch 21 , train loss: 0.0014179580977573142 , validation loss: 0.0028146475166021864
epoch 22 , train loss: 0.0014060995849718091 , validation loss: 0.0029228399826267512
epoch 23 , train loss: 0.001395757855725479 , validation loss: 0.002829206611840088
epoch 24 , train loss: 0.0013853934224036803 , validation loss: 0.002690992334400327
epoch 25 , train loss: 0.0013748576238154866 , validation loss: 0.0027592585087562187
epoch 26 , train loss: 0.0013619716837432693 , validation loss: 0.0026968742221499717
epoch 27 , train loss: 0.0013547799155404405 , validation loss: 0.0026798711873638015
epoch 28 , train loss: 0.0013488534017779103 , validation loss: 0.002697249237676605
epoch 29 , train loss: 0.0013407245557110242 , validation loss: 0.002667812662642947
epoch 30 , train loss: 0.001337291529787552 , validation loss: 0.0027169116489448005
epoch 31 , train loss: 0.0013264600756845543 , validation loss: 0.002639400440315289
epoch 32 , train loss: 0.0013211711728166584 , validation loss: 0.0026854740268305376
epoch 33 , train loss: 0.0013168124985212626 , validation loss: 0.0026303854669252618
epoch 34 , train loss: 0.0013124150531291677 , validation loss: 0.002638007269244278
epoch 35 , train loss: 0.0013073354402910662 , validation loss: 0.002623214909894264
epoch 36 , train loss: 0.0013007580146280236 , validation loss: 0.002620082107433101
epoch 37 , train loss: 0.0012962312361962146 , validation loss: 0.002596503323942611
epoch 38 , train loss: 0.001293499391643797 , validation loss: 0.002633200128250066
epoch 39 , train loss: 0.0012899871479285128 , validation loss: 0.0025909644490215485
epoch 40 , train loss: 0.0012857729238109066 , validation loss: 0.002600317955057436
epoch 41 , train loss: 0.0012813387454951159 , validation loss: 0.002597950076741513
epoch 42 , train loss: 0.0012782597367641537 , validation loss: 0.002566198549327518
epoch 43 , train loss: 0.0012766182933517055 , validation loss: 0.002570360149962942
epoch 44 , train loss: 0.001272755096706518 , validation loss: 0.0025682188843676166
epoch 45 , train loss: 0.0012696451851239724 , validation loss: 0.002580938360062075
epoch 46 , train loss: 0.0012663087131446827 , validation loss: 0.0025799278408584828
epoch 47 , train loss: 0.001264513473620393 , validation loss: 0.0025612633326511144
epoch 48 , train loss: 0.0012628170242900182 , validation loss: 0.0025802823718274115
epoch 49 , train loss: 0.0012602980342479339 , validation loss: 0.0025817765956845417
epoch 50 , train loss: 0.0012583210465236775 , validation loss: 0.002543015826059781

Test_loss: 0.0023774674480834756
RMSE: 2.816279145561854 MAPE: 3.2111769191429413 MAE: 1.5587962842430763
'''