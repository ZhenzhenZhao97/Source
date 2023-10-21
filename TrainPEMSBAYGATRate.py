import random
import torch.nn as nn
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from GATmodel import GATTCNModel


torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "./datasets/pemsbay/Speed_325.csv"
weight_path = "./datasets/pemsbay/adj_mx_bay.npy"
save_path = "./pretrain_model/pemsbay/model_pemsbay_gat_rate2.pt"
n_train, n_val, n_test = 70, 7, 7
day_slot = 288
node_nums = 325
missing_rate = 0.4
n_his = 12
kernel_size = 2
n_pred = 0
batch_size = 64
epochs = 50
embedding_size = 8 #32  #16
num_channels = [16, 32, 16]
input_feature = 1
lr = 1e-3
Ks = 3
n_heads = 3
W = get_adj_bay_matrix(weight_path, node_nums)
# print(W)
#L = scaled_laplacian(W)
#Lk = cheb_poly(L, Ks)
Lk = torch.Tensor(W.astype(np.float32)).to(device)
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

model = GATTCNModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
                         num_channels=num_channels, Lk=Lk, kernel_size=kernel_size,n_heads=n_heads).to(device)
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

# best_model = GATTCNModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
#                          num_channels=num_channels, Lk=Lk, kernel_size=kernel_size,n_heads=n_heads).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model_2(best_model, test_iter)
# print("Test_loss:", l)
# RMSE, MAPE, MAE = evaluate_metric(best_model, test_iter, data_path)
# print("RMSE:", RMSE, "MAPE:", MAPE, "MAE:", MAE)

'''
epoch 1 , train loss: 0.08466091482876548 , validation loss: 0.01776683898935649
epoch 2 , train loss: 0.011566158165735427 , validation loss: 0.013469677188881837
epoch 3 , train loss: 0.009549203394398732 , validation loss: 0.010755183085628226
epoch 4 , train loss: 0.008285281568277644 , validation loss: 0.008844576183084039
epoch 5 , train loss: 0.007420695755561567 , validation loss: 0.008385246396958135
epoch 6 , train loss: 0.007010688813755797 , validation loss: 0.0077959478219424465
epoch 7 , train loss: 0.006693455367000863 , validation loss: 0.00728149052390895
epoch 8 , train loss: 0.006272418073572069 , validation loss: 0.006509991104163656
epoch 9 , train loss: 0.005711720098826367 , validation loss: 0.006132182013027297
epoch 10 , train loss: 0.005330702565541244 , validation loss: 0.005873433586840181
epoch 11 , train loss: 0.005057207124654536 , validation loss: 0.005702823727108757
epoch 12 , train loss: 0.004870786973097049 , validation loss: 0.0056796075885249915
epoch 13 , train loss: 0.004696929332948625 , validation loss: 0.005270494447248714
epoch 14 , train loss: 0.004509324996428138 , validation loss: 0.005184598544337789
epoch 15 , train loss: 0.004319504001996777 , validation loss: 0.005046110567091635
epoch 16 , train loss: 0.004126577048123943 , validation loss: 0.004937618904588086
epoch 17 , train loss: 0.0039433000374406715 , validation loss: 0.0049932677283894625
epoch 18 , train loss: 0.003762134561188377 , validation loss: 0.005035572930019639
epoch 19 , train loss: 0.003607182425474186 , validation loss: 0.004751127253705851
epoch 20 , train loss: 0.0034719941007187605 , validation loss: 0.005044078471666455
epoch 21 , train loss: 0.003371431219452524 , validation loss: 0.004908914663389425
epoch 22 , train loss: 0.0032913538351628108 , validation loss: 0.00474137783806066
epoch 23 , train loss: 0.003219905732869103 , validation loss: 0.004979272331473768
epoch 24 , train loss: 0.0031511186411707073 , validation loss: 0.0047751804712877375
epoch 25 , train loss: 0.00308421676706613 , validation loss: 0.004873100359712321
epoch 26 , train loss: 0.003026163773928193 , validation loss: 0.004786363943341146
epoch 27 , train loss: 0.002977174029879122 , validation loss: 0.0046831886888044115
epoch 28 , train loss: 0.002926107942087786 , validation loss: 0.004590178087695688
epoch 29 , train loss: 0.002874972347516899 , validation loss: 0.004436502988196145
epoch 30 , train loss: 0.002819404832162866 , validation loss: 0.004538754803138627
epoch 31 , train loss: 0.002766993203761883 , validation loss: 0.0044431327684874185
epoch 32 , train loss: 0.0027247706828509226 , validation loss: 0.004530696008381322
epoch 33 , train loss: 0.0026871626744216837 , validation loss: 0.004422374496566898
epoch 34 , train loss: 0.002647903395404577 , validation loss: 0.004596800670049827
epoch 35 , train loss: 0.002610185857172508 , validation loss: 0.004554656322813982
epoch 36 , train loss: 0.002577658162426824 , validation loss: 0.0043094670443251425
epoch 37 , train loss: 0.002555835596561939 , validation loss: 0.004475337113846956
epoch 38 , train loss: 0.002528265368144396 , validation loss: 0.004424015925707254
epoch 39 , train loss: 0.002505057416289522 , validation loss: 0.004380518117128973
epoch 40 , train loss: 0.0024812031930668906 , validation loss: 0.004438969264020197
epoch 41 , train loss: 0.002461452229508516 , validation loss: 0.004428763269884353
epoch 42 , train loss: 0.0024450522214664186 , validation loss: 0.00439931749922904
epoch 43 , train loss: 0.0024297989812660857 , validation loss: 0.0043636312946732925
epoch 44 , train loss: 0.0024119816832398095 , validation loss: 0.004449093035317078
epoch 45 , train loss: 0.002395260993049294 , validation loss: 0.004531879499991717
epoch 46 , train loss: 0.002379842564490513 , validation loss: 0.004427132642572883
epoch 47 , train loss: 0.0023664879923625867 , validation loss: 0.00437421308022978
epoch 48 , train loss: 0.002355227302020207 , validation loss: 0.0044244126416262495
epoch 49 , train loss: 0.0023415031777084825 , validation loss: 0.004394213046159071
epoch 50 , train loss: 0.0023320868230689473 , validation loss: 0.004490991221844589
Test_loss: 0.00437496688715007
RMSE: 4.032035466475956 MAPE: 4.533027568355476 MAE: 2.4050721351864817

'''
# best_model = AutoEncoderModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
#                          num_channels=num_channels, Lk=Lk, kernel_size=kernel_size).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model_2(best_model, test_iter)
# RMSE = evaluate_metric(best_model, test_iter)
# print("RMSE:", RMSE)
# #
