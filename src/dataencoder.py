from imports import *

from utils.obj_det_utils import meshgrid, box_iou, box_nms, change_box_order

class DataEncoder():
    def __init__(self):
        self.anchor_areas = [32*32, 64*64, 128*128, 256*256, 512*512] #p3->p7
        self.aspect_ratios = [1/2, 1/1, 2/1]
        self.scale_ratios = [1, pow(2, 1/3), pow(2,2/3)]
        self.anchors_wh = self._get_anchors_wh()
    
    def _get_anchors_wh(self):
        """ Compute anchors width and height for each feature map
        Arguments:
            N/A
        Returns:
            (tensor) sized [#fm, #anchors_per_cell, 2]
        """
        anchors_wh = []
        for s in self.anchor_areas:
            for asp_ratio in self.aspect_ratios: #w/h = ar
                h = math.sqrt(s/asp_ratio) #TODO: is it sqrt(s)/asp_ratio?
                w = asp_ratio * h
                for sr in self.scale_ratios:
                    anchor_h, anchor_w = h*sr, w*sr
                    anchors_wh.append([anchor_w, anchor_h])
        num_feat_maps = len(self.anchor_areas)
        return torch.Tensor(anchors_wh).view(num_feat_maps, -1, 2)
    
    def _get_anchor_boxes(self, input_size):
        """ Compute anchor boxes for each feature map

        Args:
            input_size: (tensor) model input size of (w,h)
        Returns:
            boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
                    where #anchors = fmw*fmh*#anchors_per_cell
        """
        num_feat_maps = len(self.anchor_areas)
        fm_sizes = [(input_size/pow(2,i+3)).ceil() for i in range(num_feat_maps)]

        boxes = []
        for i in range(num_feat_maps):
            fm_size = fm_sizes[i]
            grid_size = input_size/fm_size
            fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
            xy = meshgrid(fm_w, fm_h).float() + 0.5 # 0.5 for centering the mesh
            xy = (xy*grid_size)
            xy = xy.view(fm_h, fm_w, 1, 2).expand(fm_h, fm_w, 9, 2)
            wh = self.anchors_wh[i].view(1,1,9,2).expand(fm_h, fm_w, 9, 2)
            box = torch.cat([xy,wh], 3)
            boxes.append(box.view(-1,4))
        return torch.cat(boxes,0)

    def encode(self, boxes, labels, input_size):

        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes = self._get_anchor_boxes(input_size)
        boxes = change_box_order(boxes, 'xyxy2xywh')

        ious = box_iou(anchor_boxes, boxes, order='xywh')
        max_ious, max_ids = ious.max(1)
        boxes = boxes[max_ids]

        loc_xy = (boxes[:,:2]-anchor_boxes[:,:2]) / anchor_boxes[:, 2:]
        loc_wh = torch.log(boxes[:,2:]/ anchor_boxes[:, 2:])
        loc_targets = torch.cat([loc_xy, loc_wh], 1)
        cls_targets = 1+ labels[max_ids]

        cls_targets[max_ious<0.5] = 0
        ignore = (max_ious>0.4) & (max_ious<0.5)
        cls_targets[ignore]= -1
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, input_size):
        CLS_TRESH = 0.5
        NMS_TRESH = 0.5

        input_size = torch.Tensor([input_size, input_size]) if isinstance(input_size, int) else torch.Tensor(input_size)
        anchor_boxes =self._get_anchor_boxes(input_size)

        loc_xy = loc_preds[:, :2]
        loc_wh = loc_preds[:, 2:]

        xy = loc_xy*anchor_boxes[:, 2:] + anchor_boxes[:, 2:]
        wh = loc_wh.exp() * anchor_boxes[:, 2:]
        boxes = torch.cat([xy-wh/2, xy+ wh/2], 1)

        score, labels = cls_preds.sigmoid().max(1)
        ids = score>CLS_TRESH
        ids = ids.nonzero().squeeze()
        keep = box_nms(boxes[ids], score[ids], threshold=NMS_TRESH)
        return boxes[ids][keep], labels[ids][keep]


