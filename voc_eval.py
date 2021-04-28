from utils import compute_ap
import numpy as np
import itertools

def evaluate(model, dataloader, device):
    AP = []
    model.eval()
    for images, targets in dataloader:
        
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        preds = model(images, targets)
        
        for gt, pred in zip(targets,preds):
            gt_boxes = gt['boxes'].cpu().numpy()
            gt_labels = gt['labels'].cpu().numpy()
            boxes = pred['boxes'].detach().cpu().numpy()
            labels = pred['labels'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()
        
            AP.append(compute_ap(gt_boxes,gt_labels,boxes,labels,scores))
     
    return np.mean(AP)
