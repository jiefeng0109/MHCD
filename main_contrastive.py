import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import train_contrastive
from CSANet_contrastive import Finalmodel
import sys
from scipy.io import savemat
sys.path.append('../global_module/')


from generate_pic_contrastive import aa_and_each_accuracy, sampling,load_dataset, generate_png, generate_iter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')


global Dataset  # UP,IN,KSC
Dataset = 'china'
T11, T22, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset()
print(T11.shape)

image_x, image_y, BAND1 = T11.shape



data1 = T11.reshape(np.prod(T11.shape[:2]), np.prod(T11.shape[2:]))
data2 = T22.reshape(np.prod(T22.shape[:2]), np.prod(T22.shape[2:]))
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
CLASSES_NUM = 2


print('-----Importing Setting Parameters-----')
ITER = 1
PATCH_LENGTH = 4
# number of training samples per class
# lr, num_epochs, batch_size = 0.00002, 100, 32
lr, num_epochs, batch_size = 0.00002, 100,32
loss = torch.nn.CrossEntropyLoss()


img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
img_channels1 = T11.shape[2]
img_channels2 = T22.shape[2]
INPUT_DIMENSION1 = T11.shape[2]
INPUT_DIMENSION2 = T22.shape[2]
ALL_SIZE = T11.shape[0] * T22.shape[1]
VAL_SIZE = int(TRAIN_SIZE)/2



KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

data1 = preprocessing.scale(data1)
data2 = preprocessing.scale(data2)

data1_ = data1.reshape(T11.shape[0], T11.shape[1], T11.shape[2])
whole_data1 = data1_
data2_ = data2.reshape(T22.shape[0], T22.shape[1], T22.shape[2])
whole_data2 = data2_


padded_data1 = np.lib.pad(whole_data1, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)
padded_data2 = np.lib.pad(whole_data2, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

for index_iter in range(ITER):
    print(index_iter)
    net = Finalmodel()
    params_to_optimize=[
    {"params": net.siamesnet.parameters()},  # 其他网络部分的参数
    {"params": net.chnet.parameters()},  # 其他网络部分的参数
    {"params": net.fc.parameters()},  # 其他网络部分的参数
    {"params": net.bf.parameters()},  # self.bf = BSNET_Conv() 的参数
    {"params": net.conv_con.parameters()},
    {"params": net.fc_con.parameters()}
    ]
    # optimizer = optim.Adam(params_to_optimize, lr=lr)  # , weight_decay=0.0001)
    optimizer = optim.Adam(params_to_optimize[0:2], lr=lr)  # , weight_decay=0.0001)
    optimizer_bs = optim.Adam([{'params': net.bf.parameters()},], lr=0.00001)  # , weight_decay=0.0001)
    optimizer_con = optim.Adam(params_to_optimize[4:5], lr=0.00002)  # , weight_decay=0.0001)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')


    train_iter,  valida_iter, all_iter,train_all_1,train_all_2,train_all_label = generate_iter(TRAIN_SIZE, train_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data1, whole_data2, PATCH_LENGTH, padded_data1, padded_data2, INPUT_DIMENSION1, INPUT_DIMENSION2, batch_size, gt)
    #R
    #
    train_contrastive.train(net, train_iter, valida_iter, loss, optimizer,optimizer_bs, optimizer_con, device, num_epochs,train_all_1,train_all_2,train_all_label)

    pred_test_fdssc = []
    # tic2 = time.clock()
    newnet = torch.load("best.pth")
    with torch.no_grad():
        for X1, X2, y in all_iter:
            X1 = X1.to(device)
            X2 = X2.to(device)
            newnet.eval()  # 评估模式, 这会关闭dropout
            y_hat,_,_,_,_,_ = newnet(X1,X2)
            pred_test_fdssc.extend(np.array(y_hat.cpu().argmax(axis=1)))
    # toc2 = time.clock()

    collections.Counter(pred_test_fdssc)
    gt_test = gt[total_indices]-1
    # gt_test = gt_test1[:-VAL_SIZE]
    # print('pre', np.unique(pred_test_fdssc))
    # print('gt_test', np.unique(gt_test[:-VAL_SIZE]))

    overall_acc_fdssc = metrics.accuracy_score(pred_test_fdssc, gt_test)
    confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test)
    # print(confusion_matrix_fdssc)
    each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
    kappa = metrics.cohen_kappa_score(pred_test_fdssc, gt_test)

    # print("testing time:",toc2-tic2)
    savemat("river.mat", mdict={'result': pred_test_fdssc})
    KAPPA.append(kappa)
    OA.append(overall_acc_fdssc)
    AA.append(average_acc_fdssc)
    ELEMENT_ACC[index_iter, :] = each_acc_fdssc


    def calculate_metrics(confusion_matrix):
        # 从混淆矩阵中提取TP, TN, FP, FN
        TP = confusion_matrix[1, 1]
        TN = confusion_matrix[0, 0]
        FP = confusion_matrix[0, 1]
        FN = confusion_matrix[1, 0]

        # 计算Precision
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0

        # 计算Recall
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0

        # 计算F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 计算IoU (Intersection over Union)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0

        return precision, recall, f1, iou


    precision, recall, f1, iou = calculate_metrics(confusion_matrix_fdssc)

    print("-------- Training Finished-----------")
    print('OA:',OA)
    print('AA:',AA)
    print('Kappa:',KAPPA)
    print('OA_UN',each_acc_fdssc)
    print('precision',precision)
    print('recall',recall)
    print('f1',f1)
    print('iou',iou)


    generate_png(pred_test_fdssc, gt_hsi-1, 'China', total_indices)


