from imports import *

from utils.obj_det_utils import one_hot_embedding


class FocalLoss(nn.Module):

    def __init__(self, num_classes=500):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0
        num_pos = pos.data.long().sum()

        #localization loss
        mask = pos.unsqueeze(2).expand_as(loc_preds)
        masked_loc_preds = loc_preds[mask].view(-1,4)
        masked_loc_targets = loc_targets[mask].view(-1,4)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        #focal loss
        pos_neg = cls_targets > -1
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds) #TODO: verify is this is correct
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        loss = (loc_loss + cls_loss)/num_pos.float()
        return loss

    def focal_loss(self, x, y):
        alpha=0.25

        t = one_hot_embedding(y.data, 1+self.num_classes)
        t = t[:, 1:]
        t = t.cuda()

        xt = x*(2*t -1)
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()


