import random
import torch.nn as nn
from utils import *
from torch.utils.data import DataLoader, TensorDataset
from auto_encoder import AutoEncoderModel
from tcn import TemporalConvNet


torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "./datasets/pemsbay/Speed_325.csv"
weight_path = "./datasets/pemsbay/adj_mx_bay.npy"
save_path = "./pretrain_model/pemsbay/model_pemsbay_04_3.pt"
n_train, n_val, n_test = 70, 7, 7
day_slot = 288
node_nums = 325
missing_rate = 0.4
n_his = 12
kernel_size = 2
n_pred = 0
batch_size = 64
epochs = 50
embedding_size = 32
num_channels = [16, 32, 16]
input_feature = 1
lr = 1e-3
Ks = 3
W = get_adj_bay_matrix(weight_path, node_nums)
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
# RMSE, MAPE, MAE = evaluate_metric(best_model, test_iter, data_path)
# print("RMSE:", RMSE, "MAPE:", MAPE, "MAE:", MAE)

'''epoch 1 , train loss: 0.07725613219495392 , validation loss: 0.011145964518713929
epoch 2 , train loss: 0.00662803480823348 , validation loss: 0.0073305876833608145
epoch 3 , train loss: 0.004836104899671749 , validation loss: 0.005589851006397679
epoch 4 , train loss: 0.003804318115999898 , validation loss: 0.004422711887802255
epoch 5 , train loss: 0.003083534676330899 , validation loss: 0.0035820167850380714
epoch 6 , train loss: 0.002683128017616132 , validation loss: 0.0033593888081071776
epoch 7 , train loss: 0.0025570051054133545 , validation loss: 0.003500837436763117
epoch 8 , train loss: 0.0024951407589573586 , validation loss: 0.0032150843818028385
epoch 9 , train loss: 0.0024601181740658822 , validation loss: 0.0031784390663831866
epoch 10 , train loss: 0.00242517008892846 , validation loss: 0.0032056603129101075
epoch 11 , train loss: 0.0023722292996833485 , validation loss: 0.003139777401175932
epoch 12 , train loss: 0.0023641502908553474 , validation loss: 0.003095856432869258
epoch 13 , train loss: 0.0023562786780065284 , validation loss: 0.003127472187975271
epoch 14 , train loss: 0.0023419946050506745 , validation loss: 0.0030902601304380467
epoch 15 , train loss: 0.002327748797685023 , validation loss: 0.0030505949711004897
epoch 16 , train loss: 0.0022985126810938948 , validation loss: 0.003049832053855734
epoch 17 , train loss: 0.0022889329457214625 , validation loss: 0.0030243152815199207
epoch 18 , train loss: 0.0022639793784046845 , validation loss: 0.002975543592920812
epoch 19 , train loss: 0.002228152563808896 , validation loss: 0.002948097972191636
epoch 20 , train loss: 0.0021507962660882248 , validation loss: 0.0028194125192124607
epoch 21 , train loss: 0.002013903194409231 , validation loss: 0.002650537216195826
epoch 22 , train loss: 0.0018582441444547637 , validation loss: 0.002509311929043976
epoch 23 , train loss: 0.0017131355584874297 , validation loss: 0.002401270185912565
epoch 24 , train loss: 0.0015931425229735122 , validation loss: 0.002308029104680997
epoch 25 , train loss: 0.0014987393575703679 , validation loss: 0.002308042605736309
epoch 26 , train loss: 0.0014291884269889074 , validation loss: 0.002217604934717968
epoch 27 , train loss: 0.0013845715615221133 , validation loss: 0.0021885762416158213
epoch 28 , train loss: 0.0013387975977496459 , validation loss: 0.0021651175132541486
epoch 29 , train loss: 0.0013036050865702358 , validation loss: 0.0021432764178764916
epoch 30 , train loss: 0.0012766371272065326 , validation loss: 0.0021418534020416326
epoch 31 , train loss: 0.0012539758151728525 , validation loss: 0.0021314808271564504
epoch 32 , train loss: 0.0012397433091613044 , validation loss: 0.002121785192418019
epoch 33 , train loss: 0.0012272141454405141 , validation loss: 0.002113535105042561
epoch 34 , train loss: 0.0012160381892299286 , validation loss: 0.0021074000379054307
epoch 35 , train loss: 0.0012077813803916197 , validation loss: 0.002091065943119841
epoch 36 , train loss: 0.0011976439651950057 , validation loss: 0.002094113912565231
epoch 37 , train loss: 0.0011917466644770234 , validation loss: 0.002094344158176344
epoch 38 , train loss: 0.0011859820628286083 , validation loss: 0.002086768634940338
epoch 39 , train loss: 0.0011801386593666664 , validation loss: 0.002079723782783596
epoch 40 , train loss: 0.0011742067144335454 , validation loss: 0.002075150210478561
epoch 41 , train loss: 0.0011691346032210673 , validation loss: 0.002066741208774011
epoch 42 , train loss: 0.0011658846112251543 , validation loss: 0.0020606102728798347
epoch 43 , train loss: 0.001161681618008334 , validation loss: 0.0020610749569448705
epoch 44 , train loss: 0.0011582295365065774 , validation loss: 0.0020711501764182566
epoch 45 , train loss: 0.0011541652607492806 , validation loss: 0.0020645351959726246
epoch 46 , train loss: 0.0011506681765783664 , validation loss: 0.002057105324634789
epoch 47 , train loss: 0.0011480288563008973 , validation loss: 0.002055283165030093
epoch 48 , train loss: 0.001145547723438199 , validation loss: 0.002043507807581172
epoch 49 , train loss: 0.0011428560561016817 , validation loss: 0.002049472937887547
epoch 50 , train loss: 0.0011403029025909349 , validation loss: 0.002043160986284475

Test_loss: 0.0020444084447220185
RMSE: 0.045215135128870494 MAPE: 6.051995055339827 MAE: 0.026752555374941794

Test_loss: 0.0020444084447220185
RMSE: 2.7477353126936523 MAPE: 3.37783460016622 MAE: 1.572677295624351
'''

# best_model = AutoEncoderModel(input_feature=input_feature, embedding_size=embedding_size, num_inputs=n_his,
#                          num_channels=num_channels, Lk=Lk, kernel_size=kernel_size).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model_2(best_model, test_iter)
# RMSE = evaluate_metric(best_model, test_iter)
# print("RMSE:", RMSE)
# '''RMSE: 0.04853471828135024'''