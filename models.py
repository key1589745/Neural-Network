import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def get_model(backbone,num_classes):
    # get the model using our helper function
    # load a pre-trained model for classification and return
    # only the features
    
    if backbone == 'MobileNet':
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
                   box_roi_pool=roi_pooler)
    elif backbone == 'ResNet':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)
    else:
        raise NotImplementedErorr
        
    return model