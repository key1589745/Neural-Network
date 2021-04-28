import torch
import torchvision
import transforms as T

class PrepareInstance(object):
    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )
    def __call__(self, image, target):
        anno = target['annotation']
        h, w = anno['size']['height'], anno['size']['width']
        boxes = []
        classes = []
        area = []
        iscrowd = []
        objects = anno['object']
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            bbox = obj['bndbox']
            bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
            boxes.append(bbox)
            classes.append(self.CLASSES.index(obj['name']))
            iscrowd.append(int(obj['difficult']))
            area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        classes = torch.as_tensor(classes)
        area = torch.as_tensor(area)
        iscrowd = torch.as_tensor(iscrowd)

        image_id = anno['filename'][5:-4]
        image_id = torch.as_tensor([int(image_id)])

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target



def get_voc(root, image_set, train=True):
    t = [PrepareInstance()]

    if train:
        t.append(T.ToTensor())
        t.append(T.RandomHorizontalFlip(0.5))
    transforms = T.Compose(t)

    dataset = torchvision.datasets.VOCDetection(root, '2007', image_set, transforms=transforms, download=False)

    return dataset