import random
import torch.nn as nn
from utils import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from GATmodel import GATTCNModel


torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)
random.seed(2333)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
data_path = "./datasets/pemsd4/Speed_307.csv"
weight_path = "./datasets/pemsd4/distance.csv"
save_path = "./pretrain_model/pemsd4/model_pemsd4_gat_rate2.pt"
n_train, n_val, n_test = 49, 5, 5
day_slot = 288
node_nums = 307
missing_rate = 0.4
n_his = 12
n_pred = 0
batch_size = 64
epochs = 50
kernel_size = 2
embedding_size = 32
num_channels = [16, 32, 16]
input_feature = 1
lr = 1e-3
Ks = 3
n_heads = 3
W = get_adjacency_matrix(weight_path, node_nums)  # 对角线元素不是1的邻接矩阵
#print(W)
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
        # print("Test Shape")
        # print(label_y.shape)
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
#                           num_channels=num_channels, Lk=Lk, kernel_size=kernel_size,n_heads=n_heads).to(device)
# best_model.load_state_dict(torch.load(save_path))
# l = evaluate_model_2(best_model, test_iter)
# print("Test_loss:", l)
# RMSE, MAPE, MAE = evaluate_metric(best_model, test_iter,data_path)
# print("RMSE:", RMSE, "MAPE:", MAPE, "MAE:", MAE)

'''
epoch 1 , train loss: 0.10557614727874678 , validation loss: 0.01565827827443392
epoch 2 , train loss: 0.011319562093239692 , validation loss: 0.014338340136871441
epoch 3 , train loss: 0.010579573911189504 , validation loss: 0.013357521313837719
epoch 4 , train loss: 0.009909112053496313 , validation loss: 0.013267066033846205
epoch 5 , train loss: 0.008665488552347958 , validation loss: 0.009838245072425596
epoch 6 , train loss: 0.007546519555627931 , validation loss: 0.00876684274847219
epoch 7 , train loss: 0.0070384416555545155 , validation loss: 0.008535873047025733
epoch 8 , train loss: 0.006500410319435709 , validation loss: 0.007228382820740934
epoch 9 , train loss: 0.005982357636622435 , validation loss: 0.006938014074232066
epoch 10 , train loss: 0.005683654129018259 , validation loss: 0.006464846111961812
epoch 11 , train loss: 0.005423403132339374 , validation loss: 0.006339027323938287
epoch 12 , train loss: 0.0052196242144814525 , validation loss: 0.0063645633966376205
epoch 13 , train loss: 0.005031092106049378 , validation loss: 0.0057079299235386965
epoch 14 , train loss: 0.004870426543973672 , validation loss: 0.005794483930249076
epoch 15 , train loss: 0.004682385493157715 , validation loss: 0.005237464101595085
epoch 16 , train loss: 0.004536546220267237 , validation loss: 0.005030331020462007
epoch 17 , train loss: 0.004421477057995556 , validation loss: 0.005036462382463879
epoch 18 , train loss: 0.00431587372144939 , validation loss: 0.004985813254201348
epoch 19 , train loss: 0.004217332545989009 , validation loss: 0.005043465452981501
epoch 20 , train loss: 0.004113310802428635 , validation loss: 0.004811381097102961
epoch 21 , train loss: 0.004026997677210886 , validation loss: 0.004695206318749956
epoch 22 , train loss: 0.003968150672681697 , validation loss: 0.00498294569383836
epoch 23 , train loss: 0.003917721106121115 , validation loss: 0.004681061684164061
epoch 24 , train loss: 0.00387382372193931 , validation loss: 0.004439736904394863
epoch 25 , train loss: 0.0038436198785351044 , validation loss: 0.004760984187468295
epoch 26 , train loss: 0.0038070852199700673 , validation loss: 0.004609984995810725
epoch 27 , train loss: 0.003784353139073399 , validation loss: 0.004448404722420048
epoch 28 , train loss: 0.003755274297676254 , validation loss: 0.004552745558355098
epoch 29 , train loss: 0.0037349505155776616 , validation loss: 0.004611604850544234
epoch 30 , train loss: 0.003707686691062375 , validation loss: 0.004455763497642016
epoch 31 , train loss: 0.0036823881465420463 , validation loss: 0.004468457401639341
epoch 32 , train loss: 0.0036626878219006786 , validation loss: 0.004495924373496902
epoch 33 , train loss: 0.003643338971214894 , validation loss: 0.004491326197921315
epoch 34 , train loss: 0.0036215877880999025 , validation loss: 0.0044304385433303
epoch 35 , train loss: 0.00360115861192462 , validation loss: 0.00446886638152524
epoch 36 , train loss: 0.0035804124929128614 , validation loss: 0.004297720829179571
epoch 37 , train loss: 0.0035652116932443395 , validation loss: 0.004342358302499843
epoch 38 , train loss: 0.003548413533548599 , validation loss: 0.004353387831350526
epoch 39 , train loss: 0.0035297877113320084 , validation loss: 0.0041871739292537475
epoch 40 , train loss: 0.0035120851373058293 , validation loss: 0.004323010913698197
epoch 41 , train loss: 0.003493325057886401 , validation loss: 0.0042766308944860625
epoch 42 , train loss: 0.0034831583429645843 , validation loss: 0.004268343917796865
epoch 43 , train loss: 0.003469505779574032 , validation loss: 0.004192407212021764
epoch 44 , train loss: 0.003453437966156463 , validation loss: 0.004210776734646638
epoch 45 , train loss: 0.003438548879906648 , validation loss: 0.0042537477627289
epoch 46 , train loss: 0.0034258027482718277 , validation loss: 0.0042416516667911075
epoch 47 , train loss: 0.0034146603105566008 , validation loss: 0.004244512616276795
epoch 48 , train loss: 0.003407374293879822 , validation loss: 0.004213362793851684
epoch 49 , train loss: 0.0033977766278580875 , validation loss: 0.004153488591373882
epoch 50 , train loss: 0.003389930762359479 , validation loss: 0.004239802620632554
Test_loss: 0.0039173817104031245
RMSE: 3.61741361462383 MAPE: 4.055364028337022 MAE: 2.0457680126489826
'''