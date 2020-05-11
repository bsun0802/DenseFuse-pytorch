import shutil
import torch
from datetime import datetime
import numpy as np


class Logger:
    interval = 10
    TRAIN_INFO = ('[TRAIN {}] - EPOCH {:d}/{:d}, BATCH {:d}/{:d}, '  # first blank is datetime
                  'LOSS | LOSS(AVG) [PIXEL, SSIM, TOTAL]: {}({}) | {}({})')

    def __init__(self, n):
        self.val = np.zeros(n)
        self.sum = np.zeros(n)
        self.count = 0
        self.avg = np.zeros(n)

        self.val_losses = []

    def update(self, losses):
        self.val = np.array(losses)  # log the loss of current batch
        self.sum += self.val
        self.count += 1
        self.avg = self.sum / self.count  # averaged loss of batches seen so far

    def now(self):
        return datetime.now().strftime(r'%m/%d %H:%M:%S')


def save_ckpt(state, ckptPath, is_best, epoch, it):
    fp = ckptPath.joinpath(f'Epoch_{epoch}_iter_{it}_ckpt.pth')
    torch.save(state, fp)
    if is_best:
        print(f'[BEST MODEL] Obtained on epoch={epoch}, iter={it}')
        shutil.copy(fp, ckptPath.joinpath('best_model.pth'))
