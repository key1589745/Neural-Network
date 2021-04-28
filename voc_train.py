import os
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import copy
import utils
import transforms as T

from torchvision.datasets import VOCDetection


import torch
from tqdm import tqdm
import torchvision
from voc_eval import evaluate
from voc_dataset import get_voc
from  torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 21 # 20 classes + background for VOC
  
    # dataset = get_voc('.', 'trainval', transforms = get_transform(train=True) )
    dataset = get_voc('./data', 'train')
    dataset_test = get_voc('./data', 'val')
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=2, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)
    print('data prepared, train data: %d' % len(dataset))

    # get the model using our helper function
    # load a pre-trained model for classification and return
    # only the features
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler).to(device)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes).to(device)


    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 20

    # setup log data writer
    if not os.path.exists('log'):
        os.makedirs('log')
    writer = SummaryWriter(log_dir='log')
    max_AP = 0.0
    best_model = copy.deepcopy(model.state_dict())
    for epoch in range(10):
        loss_epoch = {}
        loss_name = ['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']
        for ii, (images, targets) in tqdm(enumerate(data_loader)):   
            model.train()
            model.zero_grad()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # training
            loss_dict = model(images, targets)          
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            if ii % 50 == 1:
                for key, value in loss_dict.items():
                    writer.add_scalar(key, value, ii)
            if ii % 200 == 1:
                mAP = evaluate(model, data_loader_test, device=device)
                writer.add_scalar('mAP', mAP, ii)
                if mAP > max_AP:
                    max_AP = mAP
                    best_model = copy.deepcopy(model.state_dict())
                    
    torch.save(best_model, 'my_model.pth')
    writer.close()
