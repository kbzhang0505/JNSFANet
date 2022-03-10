import numpy as np
import random
from scipy import stats
from argparse import ArgumentParser
import yaml
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics.metric import Metric
from torchvision import models
from Dataset import IQADataset
import matplotlib as plt
import time
from tensorboardX import SummaryWriter
import torch.nn as nn
# from torchsummary import summary
import os
import shutil

start_time = time.time()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def loss_fn(y_pred, y):
    loss_1 = F.l1_loss(y_pred, y)
    return loss_1


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


# ==============Performance metric setting=================
class IQAPerformance(Metric):
    """
    Evaluation of IQA methods using SROCC, KROCC, PLCC, RMSE, MAE, OR.

    `update` must receive output of the form (y_pred, y).
    """

    def reset(self):
        self._y_pred = []
        self._y = []
        # self._y_pred_earth = []

    def update(self, output):
        y_pred, y = output

        self._y.append(y[0].item())

        self._y_pred.append(torch.mean(y_pred).item())

        # self._y_pred_earth.append(torch.mean(y_pred[1].item()))

    def compute(self):
        sq = np.reshape(np.asarray(self._y), (-1,))

        q = np.reshape(np.asarray(self._y_pred), (-1,))
        # print("mos:{}".format(list(sq)))
        # print("pre {}".format(list(q)))
        # q_2 = np.reshape(np.asarray(self._y_pred_earth), (-1,))

        srocc = stats.spearmanr(sq, q)[0]
        krocc = stats.stats.kendalltau(sq, q)[0]
        plcc = stats.pearsonr(sq, q)[0]
        rmse = np.sqrt(((sq - q) ** 2).mean())
        mae = np.abs((sq - q)).mean()

        return srocc, krocc, plcc, rmse, mae  # , srocc_2, krocc_2, plcc_2, rmse_2


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                    stride=1, padding=0, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out + x
        return out


class JNSFANet(nn.Module):
    cfg = [(6, 32, 1, 1),
           (3, 50, 2, 1),
           (3, 32, 1, 1)]  # [(expand，out_channels，block_num，stride)]

    def __init__(self, ker_size=7, n_kers=50, n1_nodes=800, n2_nodes=800):
        super(JNSFANet, self).__init__()
        self.ca = ChannelAttention(in_planes=6, ratio=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer = self._make_layers(in_planes=32)

        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.conv2 = nn.Conv2d(96, 100, 1)
        # self.conv3 = nn.Conv2d(36, 36, 1)
        self.linear = nn.Sequential(
            # nn.Linear(136, 800), nn.ReLU(), nn.Dropout(),
            # nn.Linear(800, 800), nn.ReLU(),
            nn.Linear(136, 1))

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x_rgb = x[0].view(-1, x[0].size(-3), x[0].size(-2), x[0].size(-1))  # [512,3,32,32]
        x_feature = x[1].unsqueeze(-1).unsqueeze(-2)  # [2,2,128,3,32,32,1,1]
        x_feature = x_feature.view(-1, x_feature.size(-3), x_feature.size(-2), x_feature.size(-1))  # [49152,32,1,1]
        x_fusion = x[2].view(-1, x[2].size(-3), x[2].size(-2), x[2].size(-1))

        attention = self.ca(x_fusion)

        out = F.relu(self.bn1(self.conv1(torch.mul(x_rgb, attention))))  # [512,32,32]
        out = self.layer(out)  # [512,50,32,32]
        out_1 = self.maxpool(out)  # [512,50,1,1]
        out_2 = -self.maxpool(-out)  # [512,50,1,1]
        out_3 = global_std_pool2d(out)
        out = torch.cat((out_1, out_2, out_3), 1)  # [512,150,1,1]
        # ==============RGB===============
        out_conv = self.conv2(out)  # [512,100,1,1]
        out = out_conv.squeeze().squeeze()  # [512,100]

        # ===============NSS==================
        x_feature = x_feature.squeeze().squeeze()
        # ===============Cat & FC==============
        out_all = torch.cat((out, x_feature,), 1)
        out = self.linear(out_all)

        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, 3, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


def get_data_loaders(config, train_batch_size, exp_id=0):
    train_dataset = IQADataset(config, exp_id, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=train_batch_size,
                                               shuffle=True,
                                               num_workers=0)

    val_dataset = IQADataset(config, exp_id, 'val')
    val_loader = torch.utils.data.DataLoader(val_dataset)

    if config['test_ratio']:
        test_dataset = IQADataset(config, exp_id, 'test')
        test_loader = torch.utils.data.DataLoader(test_dataset)

        return train_loader, val_loader, test_loader

    return train_loader, val_loader


# =================Train=======================
def run(train_batch_size, epochs, lr, weight_decay, config, exp_id, disable_gpu=False):
    # TensorboardX
    writer = SummaryWriter()
    # ======DATALODER=========
    if config['test_ratio']:  # 0.2
        train_loader, val_loader, test_loader = get_data_loaders(config, train_batch_size, exp_id)
    else:
        train_loader, val_loader = get_data_loaders(config, train_batch_size, exp_id)
    # ===========Device setting====================
    device = torch.device("cuda" if not disable_gpu and torch.cuda.is_available() else "cpu")
    # ==========Model setting===========
    model = JNSFANet()
    model = model.to(device)
    # print(model)

    # ===========Optimizer setting==============
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # ===========For training=============
    global best_criterion
    global path_checkpoint
    best_criterion = -1  # SROCC>=-1
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'IQA_performance': IQAPerformance()},
                                            device=device)

    # =========Parameters number=========
    params = params_count(model)
    print("Parameters number：{}".format(params))

    global best_epoch

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
        print('\n')
        print("Validation Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
              .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE))
        writer.add_scalar("validation/SROCC", SROCC, engine.state.epoch)
        writer.add_scalar("validation/KROCC", KROCC, engine.state.epoch)
        writer.add_scalar("validation/PLCC", PLCC, engine.state.epoch)
        writer.add_scalar("validation/RMSE", RMSE, engine.state.epoch)
        writer.add_scalar("validation/MAE", MAE, engine.state.epoch)

        global best_criterion
        global best_epoch
        if SROCC > best_criterion:
            best_criterion = SROCC
            best_epoch = engine.state.epoch
            path_checkpoint = 'NSSIQP_best_epoch_QADS.pkl'  # 修改模型参数保存地址
            torch.save(model.state_dict(), path_checkpoint)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_testing_results(engine):
        if config["test_ratio"] > 0 and config['test_during_training']:
            evaluator.run(test_loader)
            writer.add_scalar("test_pre", engine.state.output, engine.state.epoch)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
            print("Testing Results    - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
                  .format(engine.state.epoch, SROCC, KROCC, PLCC, RMSE, MAE))
            writer.add_scalar("testing/SROCC", SROCC, engine.state.epoch)
            writer.add_scalar("testing/KROCC", KROCC, engine.state.epoch)
            writer.add_scalar("testing/PLCC", PLCC, engine.state.epoch)
            writer.add_scalar("testing/RMSE", RMSE, engine.state.epoch)
            writer.add_scalar("testing/MAE", MAE, engine.state.epoch)

    @trainer.on(Events.COMPLETED)
    def final_testing_results(engine):
        if config["test_ratio"]:
            model.load_state_dict(torch.load('./JNSFANet_best_epoch_QADS.pkl'))  # model weights saving path
            evaluator.run(test_loader)
            metrics = evaluator.state.metrics
            SROCC, KROCC, PLCC, RMSE, MAE = metrics['IQA_performance']
            global best_epoch
            print('\n')
            print("Final Test Results - Epoch: {} SROCC: {:.4f} KROCC: {:.4f} PLCC: {:.4f} RMSE: {:.4f} MAE: {:.4f}%"
                  .format(best_epoch, SROCC, KROCC, PLCC, RMSE, MAE))
            path_test = "JNSFANet_best_epoch_QADS"  # model indices saving path
            np.save(path_test, (SROCC, KROCC, PLCC, RMSE, MAE))

    trainer.run(train_loader, max_epochs=epochs)

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(description='PyTorch JNSFANet')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0.0)')
    parser.add_argument('--config', default='config.yaml', type=str,
                        help='config file path (default: config.yaml)')
    parser.add_argument('--exp_id', default='0', type=str,
                        help='exp id (default: 0)')
    parser.add_argument('--database', default='QADS', type=str,  # Dataset choose
                        help='database name (default: MA)')
    parser.add_argument('--model', default='JNSFANet', type=str,
                        help='model name (default: JNSFANet)')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument("--log_dir", type=str, default="logger",
                        help="log directory for Tensorboard log output")
    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    # parser.add_argument('--multi_gpu', action='store_true',
    #                     help='flag whether to use multiple GPUs')

    args = parser.parse_args(args=[])

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

    with open(args.config, mode='r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('exp id: ' + args.exp_id)
    print('database: ' + args.database)
    print('model: ' + args.model)
    config.update(config[args.database])
    config.update(config[args.model])

    run(args.batch_size, args.epochs, args.lr, args.weight_decay, config, args.exp_id,
        args.disable_gpu)
    end_time = time.time()
    print(end_time - start_time, 's')