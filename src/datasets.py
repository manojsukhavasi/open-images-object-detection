from imports import *

from dataencoder import DataEncoder
from transforms import resize, center_crop, random_crop, random_flip

class OpenImagesDataset(data.Dataset):
    
    def __init__(self, root, list_file, transform, train=False, input_size=600):
        """
        Arguments:
            root: (str) images path
            list_file: (str) path to index file
            train: (boolean) train or test
            transform: ([transforms]) image transforms to be applied
            input_size: (int) model input image size 
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        with open(list_file) as f:
            lines = f.readlines()
            lines = lines[1:] # Removing the header
            self.num_samples = len(lines)

        for line in lines:
            splitted = line.split('\"')
            fn = splitted[0].split(',')[0] + '.jpg'
            bboxes = ast.literal_eval(splitted[1])
            self.fnames.append(fn)
            num_boxes = len(bboxes)
            box = []
            label = []
            for i in range(num_boxes):
                c, xmin, ymin, xmax, ymax = bboxes[i]
                box.append([float(xmin), float(ymin), float(xmax), float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        """
        """
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        boxes = self.boxes[idx].clone()
        # converting ratios into numbers
        w,h = img.size
        boxes = boxes * torch.Tensor([w,h,w,h])
        labels = self.labels[idx].clone()
        size = self.input_size

        #Data Augmentation
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
        else:
            img, boxes = resize(img, boxes, (size,size))
            #img, boxes = center_crop(img, boxes, (size,size))
        
        img = self.transform(img)
        return img, boxes, labels        

    def __len__(self):
        return self.num_samples

    def collate_fn(self, batch):
        ''' Pad images and encode targets 
        
        Args:
            batch, (list) of images, cls_targets, loc_targets
        Returns:
            padded_images, stacked loc_targets, stacked_cls_targets
        '''
        imgs   = [x[0] for x in batch]
        boxes  = [x[1] for x in batch]
        labels = [x[2] for x in batch] 

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)
    