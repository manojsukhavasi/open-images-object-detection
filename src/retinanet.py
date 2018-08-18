from imports import *

from fpn import FPN50

class RetinaNet(nn.Module):

    def __init__(self, num_classes=500, num_anchors=9):
        super().__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fpn_feat_maps = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fpn_feat_maps:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            # TODO: Figure out next two lines
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)
    
    def _make_head(self, out_size):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_size, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)
