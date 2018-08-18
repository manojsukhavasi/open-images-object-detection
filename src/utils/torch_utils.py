from imports import *

def save_checkpoint(state, is_best, fname='checkpoint.pth.tar'):
    torch.save(state, fname)
    if is_best:
        shutil.copyfile(fname, 'best_val_loss.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count