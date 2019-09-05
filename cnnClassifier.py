import numpy as np
import torch
import torch.nn.functional as F
import h5py
import os
from kdigo_funcs import load_csv
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
import datetime
import argparse
import json

SEED = 5

np.random.seed(SEED)
torch.manual_seed(SEED)

parser = argparse.ArgumentParser(description='Merge clusters.')
parser.add_argument('--config_file', action='store', nargs=1, type=str, dest='cfname',
                    default='kdigo_conf.json')
parser.add_argument('--config_path', action='store', nargs=1, type=str, dest='cfpath',
                    default='')
parser.add_argument('--balance_samples', '-bs', action='store_true', dest='bal_samp')
parser.add_argument('--balance_weights', '-bw', action='store_true', dest='bal_weight')
parser.add_argument('--learning_rate', '-lr', action='store', nargs=1, dest='lr', dtype=float, default=1e-2)
parser.add_argument('--learning_rate_decay', '-lr_decay', action='store', nargs=1, dest='lr_decay', dtype=float, default=0.1)
parser.add_argument('--weight_decay', '-w_decay', action='store', nargs=1, dest='w_decay', dtype=float, default=0.0)
parser.add_argument('--optimizer', '-opt', action='store', nargs=1, dest='opt', dtype=str, default='sgd',
                    choices=['adamax', 'adagrad', 'adam', 'sgd', 'rms', 'lbfgs'])
parser.add_argument('--momentum', '-mom', action='store', nargs=1, dest='momentum', dtype=float, default=0.95)
parser.add_argument('--max_epoch', '-me', action='store', nargs=1, dest='epoch', dtype=int, default=100)
parser.add_argument('--folds', '-folds', action='store', nargs=1, dest='folds', dtype=int, default=5)
parser.add_argument('--feature', '-f', action='store', nargs='*', dest='feat', dtype=str, default='max_kdigo_7d')
parser.add_argument('--target', '-t', action='store', nargs=1, dest='targ', dtype=str, default='died_inp')
parser.add_argument('--meta_group', '-meta', action='store', type=str, dest='meta',
                    default='meta')
args = parser.parse_args()

configurationFileName = os.path.join(args.cfpath, args.cfname)
fp = open(configurationFileName, 'r')
conf = json.load(fp)
fp.close()

basePath = conf['basePath']
cohortName = conf['cohortName']
t_lim = conf['analysisDays']
tRes = conf['timeResolutionHrs']
v = conf['verbose']
analyze = conf['analyze']
meta_grp = args.meta

baseDataPath = os.path.join(basePath, 'DATA', 'all_sheets')
dataPath = os.path.join(basePath, 'DATA', analyze, cohortName)
resPath = os.path.join(basePath, 'RESULTS', analyze, cohortName)

f = h5py.File(os.path.join(resPath, 'stats.h5'), 'r')
ids = f[meta_grp]['ids'][:]

balanceSamples = args.bal_samp
balanceWeights = args.bal_weight
lr = args.lr
lr_decay = args.lr_decay
wd = args.w_decay
mom = args.momentum
opt = args.opt
max_epoch = args.epoch
folds = args.folds
# adamax adagrad adam sgd rms lbfgs

# feats = ['dba_65clust_norm','kdigo_frontfill',
#          'kdigo_backfill', 'kdigo_zeropad_front', 'kdigo_zeropad_back',
#          'rel_scr_frontfill', 'rel_scr_backfill', 'rel_scr_zeropad_front', 'rel_scr_zeropad_back']
feats = ['dba_65clust_norm']
targets = ['died_inp', ]


class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1a = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=2)
        self.layer1b = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=8, stride=2)
        # self.layer1c = torch.nn.Conv1d(in_channels=1, out_channels=20, kernel_size=7, stride=2)

        self.layer2a = torch.nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, stride=1)
        self.layer2b = torch.nn.Conv1d(in_channels=32, out_channels=8, kernel_size=3, stride=1)
        # self.layer2c = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=3, stride=1)

        # Concatenate
        self.fc1 = torch.nn.Linear(in_features=88, out_features=32, bias=True)
        self.fc2 = torch.nn.Linear(in_features=32, out_features=1, bias=True)
        self.training = False

    def forward(self, x):
        xa = F.relu(self.layer1a(x))
        xa = F.relu(self.layer2a(xa))
        xa = F.dropout(F.max_pool1d(xa, kernel_size=2), training=self.training)
        xa = xa.view(xa.size()[0], -1)

        xb = F.relu(self.layer1b(x))
        xb = F.relu(self.layer2b(xb))
        xb = F.dropout(F.max_pool1d(xb, kernel_size=2), training=self.training)
        xb = xb.view(xb.size()[0], -1)

        # xc = F.relu(self.layer1c(x))
        # xc = F.relu(self.layer2c(xc))
        # xc = F.dropout(F.max_pool1d(xc, kernel_size=2), training=self.training)
        # xc = xc.view(xc.size()[0], -1)

        x = torch.cat((xa, xb), dim=1)
        # print(x.size())
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.sigmoid(self.fc2(x))
        # log_probs = torch.nn.functional.log_softmax(x, dim=1)
        log_probs = x
        return log_probs


def make_weights_for_balanced_classes(labels):
    nclasses = len(np.unique(labels))
    count = [0] * nclasses
    for item in labels:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for target in targets:
    y = f['meta'][target][:]
    for feat in feats:
        X = load_csv(os.path.join(resPath, 'features', 'individual', feat + '.csv'), ids)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        date_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d_%H%M%S')

        n_pos = np.sum(y)
        n_neg = len(y) - n_pos
        weight_dict = {0: 1 - (n_neg / len(y)), 1: 1 - (n_pos / len(y))}

        if folds > 1:
            skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        else:
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)

        filterwarnings('ignore')
        foldCount = 1
        savePath = os.path.join(resPath, 'classification', 'cnn')
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath = os.path.join(savePath, target)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath = os.path.join(savePath, feat)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        savePath = os.path.join(savePath, date_str)
        if not os.path.exists(savePath):
            os.mkdir(savePath)
        saved = False
        try:
            for train_idx, test_idx in skf.split(X, y):
                print('Training on Fold %d of %d' % (foldCount, folds))
                model = Simple1DCNN().double().to(device)
                if not saved:
                    with open(os.path.join(savePath, 'model_summary.txt'), 'w') as f:
                        f.write(repr(model))
                        f.write('\nOptimizer:\t\t%s\n' % opt)
                        f.write('LR:\t\t\t%E\n' % lr)
                        f.write('LR Decay:\t\t%E\n' % lr_decay)
                        f.write('Weight Decay:\t\t%E\n' % wd)
                        f.write('Momentum:\t\t %E\n' % mom)
                        f.write('Balanced Sampler?\t')
                        if balanceSamples:
                            f.write('Y\n')
                        else:
                            f.write('N\n')
                        f.write('Balanced Class Weights?\t')
                        if balanceWeights:
                            f.write('Y\n')
                        else:
                            f.write('N\n')
                    saved = True

                if opt == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=mom)
                elif opt == 'adagrad':
                    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=wd)
                elif opt == 'adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr, lr_decay=lr_decay, weight_decay=wd)
                elif opt == 'adamax':
                    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
                elif opt == 'lbfgs':
                    optimizer = torch.optim.LBFGS(model.parameters())
                elif opt == 'rms':
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, momentum=mom)

                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
                # scheduler = torch.optim.lr_scheduler.ReduceLROnPLateau(optimizer)
                # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
                # if epoch > 0 and epoch % 30 == 0:
                #     for param_group in optimizer.param_groups:
                #         param_group['lr'] = lr * (0.1 ** (epoch // 30))

                criterion = torch.nn.BCELoss()

                X_train, y_train = X[train_idx], y[train_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                n_pos = np.sum(y_train)
                n_neg = len(y_train) - n_pos
                class_weights = [1 - (n_neg / len(train_idx)), 1 - (n_pos / len(train_idx))]

                train_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train).double(), torch.from_numpy(y_train).double())
                test_ds = torch.utils.data.TensorDataset(torch.from_numpy(X_test).double(), torch.from_numpy(y_test).double())

                if balanceSamples:
                    train_weights = make_weights_for_balanced_classes(y_train.astype(int))
                    test_weights = make_weights_for_balanced_classes(y_test.astype(int))
                    train_weights = torch.DoubleTensor(train_weights)
                    test_weights = torch.DoubleTensor(test_weights)
                    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights))
                    test_sampler = torch.utils.data.sampler.WeightedRandomSampler(test_weights, len(test_weights))

                    train_iterator = DataLoader(train_ds, batch_size=32, sampler=train_sampler)
                    test_iterator = DataLoader(test_ds, batch_size=32, sampler=test_sampler)
                else:
                    train_iterator = DataLoader(train_ds, batch_size=32)
                    test_iterator = DataLoader(test_ds, batch_size=32)
                valid_iterator = test_iterator
                optimizer.zero_grad()
                epoch_eval = []
                epoch_train_loss = []
                epoch_test_loss = []
                print('Epoch#\tAccuracy\tPrecision\tRecall\tSpecificity\tSensitivity\tROC-AUC\t\tTrainLoss\tTestLoss')
                eval_out = open(os.path.join(savePath, 'fold%d.csv' % foldCount), 'w')
                eval_out.write('Epoch#,Accuracy,Precision,Recall,Specificity,Sensitivity,ROC-AUC,TrainLoss,TestLoss\n')
                for epoch in range(max_epoch):
                    model.train()
                    train_loss = []
                    test_loss = []
                    for tX, ty in iter(train_iterator):
                    # for tX, ty in tqdm.tqdm(iter(train_iterator), desc='Epoch %d/200' % epoch):
                        optimizer.zero_grad()
                        y_pred = model(tX)
                        if balanceWeights:
                            class_weights = torch.from_numpy(np.array([weight_dict[x.item()] for x in ty]))
                            criterion = torch.nn.BCELoss(weight=class_weights)
                        else:
                            criterion = torch.nn.BCELoss()
                        loss = criterion(y_pred, ty)
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()
                        train_loss.append(loss.item())
                    model.eval()
                    metrics = []
                    with torch.no_grad():
                        for tX, ty in iter(test_iterator):
                            y_pred = model(tX)
                            if balanceWeights:
                                class_weights = torch.from_numpy(np.array([weight_dict[x.item()] for x in ty]))
                                criterion = torch.nn.BCELoss(weight=class_weights)
                            else:
                                criterion = torch.nn.BCELoss()
                            loss = criterion(y_pred, ty).item()
                            auc = roc_auc_score(ty, y_pred)
                            y_pred = torch.round(y_pred)
                            acc = accuracy_score(ty, y_pred)
                            prec = precision_score(ty, y_pred)
                            rec = recall_score(ty, y_pred)
                            try:
                                tn, fp, fn, tp = confusion_matrix(ty, y_pred).ravel()
                                spec = tn / (tn + fp)
                                sens = tp / (tp + fn)
                            except:
                                if np.all([y_pred[idx].item() == ty[idx].item() for idx in range(len(ty))]):
                                    tn = tp = len(ty)
                                    fp = fn = 0
                                    spec = sens = 1
                                else:
                                    tn = tp = 0
                                    fp = fn = len(ty)
                                    spec = sens = 0
                            class_weights = torch.from_numpy(np.array([weight_dict[x.item()] for x in ty]))
                            criterion = torch.nn.BCELoss(weight=class_weights)

                            metrics.append((acc, prec, rec, spec, sens, auc))
                            test_loss.append(loss)
                    metrics = np.mean(metrics, axis=0)
                    epoch_eval.append(metrics)
                    train_loss = np.mean(train_loss)
                    test_loss = np.mean(test_loss)
                    epoch_test_loss.append(test_loss)
                    epoch_train_loss.append(train_loss)
                    print('%d\t\t%.2f\t\t%.2f\t\t%.2f\t%.2f\t\t%.2f\t\t%.3f\t\t%.3f\t\t%.3f' % (epoch, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], train_loss, test_loss))
                    eval_out.write('%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.3f,%.3f\n' % (
                    epoch, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], train_loss, test_loss))
                    if epoch > 20:
                        minLoss = np.where(epoch_test_loss == np.min(epoch_test_loss))[0][-1]
                        if minLoss < epoch - 10:
                            print('Stopping Early: No improvement for 10 epochs')
                            break
                    # print('Avg Train Lossa: %.2f' % train_loss)
                    # print('Avg Test Loss: %.2f' % test_loss)
                torch.save(model.state_dict(), os.path.join(savePath, 'fold%d_state_dict.pth' % foldCount))
                eval_out.close()
                fig = plt.figure()
                ax = plt.subplot(211)
                sns.lineplot(np.arange(len(epoch_train_loss)), epoch_train_loss, label='Train Loss')
                plt.xlabel('Epoch #')
                plt.ylabel('BCE Loss')
                plt.title('Training Loss')
                # plt.show()
                # plt.close(fig)
                # fig = plt.figure()
                ax = plt.subplot(212)
                sns.lineplot(np.arange(len(epoch_test_loss)), epoch_test_loss, label='Test Loss')
                plt.xlabel('Epoch #')
                plt.ylabel('BCE Loss')
                plt.title('Test Loss')
                # plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(savePath, 'fold%d_loss.png' % foldCount), dpi=600)
                # plt.close(fig)
                foldCount += 1
                if folds == 1:
                    break
        except:
            for (dirpath, dirnames, fnames) in os.walk(savePath):
                for fname in fnames:
                    os.remove(os.path.join(savePath, fname))
            os.rmdir(savePath)
