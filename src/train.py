from imports import *

from datasets import OpenImagesDataset
from retinanet import RetinaNet
from loss import FocalLoss
from utils.torch_utils import save_checkpoint, AverageMeter
import sys

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

def train(total_epochs=1, interval=100, resume=False, ckpt_path = ''):
    print("Loading training dataset...")
    train_dset = OpenImagesDataset(root='./data/train',
                            list_file ='./data/tmp/train_images_bbox.csv',
                            transform=transform, train=True, input_size=600)

    train_loader = data.DataLoader(train_dset, batch_size=4, shuffle=True, num_workers=4, collate_fn=train_dset.collate_fn)
    
    print("Loading completed.")

    #val_dset = OpenImagesDataset(root='./data/train',
    #                  list_file='./data/tmp/train_images_bbox.csv', train=False, transform=transform, input_size=600)
    #val_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=False, num_workers=4, collate_fn=val_dset.collate_fn)

    net = RetinaNet()
    net.load_state_dict(torch.load('./model/net.pth'))

    criterion = FocalLoss()
    
    net.cuda()
    criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    best_val_loss = 1000

    start_epoch=0

    if resume:
        if os.path.isfile(ckpt_path):
            print(f'Loading from the checkpoint {ckpt_path}')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f'Loaded checkpoint {ckpt_path}, epoch : {start_epoch}')
        else:
            print(f'No check point found at the path {ckpt_path}')

    

    for epoch in range(start_epoch, total_epochs):
        train_one_epoch(train_loader, net, criterion, optimizer, epoch, interval)
        val_loss = 0
        #val_loss = validate(val_loader, net, criterion, interval)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch+1,
                'state_dict': net.state_dict(),
                'best_val_loss': best_val_loss,
                'optimizer' : optimizer.state_dict()
            }, is_best=True)


def train_one_epoch(train_loader, model, loss_fn, opt, epoch, interval):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    train_loss = 0
    no_of_batches = int(train_loader.dataset.num_samples/train_loader.batch_size) + 1

    end = time.time()

    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        loc_targets = loc_targets.cuda()
        cls_targets = cls_targets.cuda()

        opt.zero_grad()
        loc_preds, cls_preds = model(inputs)
        loss = loss_fn(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        opt.step()

        batch_time.update(time.time() - end)
        end = time.time()

        train_loss += loss.data[0]
        if(batch_idx%interval == 0):
            print(f'Train -> Batch : [{batch_idx}/{no_of_batches}]| Batch avg time :{batch_time.avg} \
            | Data_avg_time: {data_time.avg} | avg_loss: {train_loss/(batch_idx+1)}')
        
        if(batch_idx%(5000)==0):
            save_checkpoint({
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_val_loss': train_loss/(batch_idx+1),
                'optimizer' : optimizer.state_dict()
            }, is_best=True, fname=f'checkpoint_{epoch}_{batch_idx}.pth.tar')

def validate(val_loader, model, loss_fn, interval):
    model.eval()
    val_loss = 0.0

    batch_time = AverageMeter()
    data_time = AverageMeter()
    no_of_batches = int(val_loader.dataset.num_samples/val_loader.batch_size) + 1

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            loc_preds, cls_preds = model(inputs)
            loss = loss_fn(loc_preds, loc_targets, cls_preds, cls_targets)

            val_loss += loss
            batch_time.update(time.time() - end)
            end = time.time()
            if(batch_idx%interval == 0):
                print(f'Val -> Batch : [{batch_idx}/{no_of_batches}]| Batch avg time :{batch_time.avg} \
                | Data_avg_time: {data_time.avg}| avg_loss: {val_loss/(batch_idx+1)}')
            

    val_loss = val_loss/len(val_loader)
    print("_________________________________________________________________________________")
    print(f'Val -> Final Loss:{val_loss} \t')
    print("_________________________________________________________________________________")

    return val_loss


if __name__=='__main__':
    sys.stdout = open('output.txt','wt')
    train()