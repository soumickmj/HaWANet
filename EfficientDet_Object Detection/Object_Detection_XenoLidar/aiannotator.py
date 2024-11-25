import torch
from torch.backends import cudnn
import cv2
import numpy as np
from aiannotator.objectdetection.efficientnet.backbone import EfficientDetBackbone
from aiannotator.objectdetection.efficientnet.efficientdet.utils import BBoxTransform, ClipBoxes
from aiannotator.objectdetection.efficientnet.utils.utils import preprocess, invert_affine, postprocess, preprocess_video

class ObjectDetector:
    def __init__(self) -> None:
        # EfficientDet configuration

        self.compound_coef = 7
        self.force_input_size = None  # set None to use default size

        self.threshold = 0.3
        self.iou_threshold = 0.3

        # Gets the GPU if there is one, otherwise the cpu
        self.use_cuda = torch.cuda.is_available()
        self.use_float16 = False
        
        cudnn.fastest = True
        cudnn.benchmark = True

        # list of object categories that can be detected by the AI object detection
        self.obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                    'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                    'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                    'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                    'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                    'toothbrush']

        # list of object categories allowed to be outputted
        self.out_class_id_dict = {
            'person'     : 1,
            'bicycle'    : 2,
            'motorcycle' : 3,
            'car'        : 4,
            'truck'      : 6,
            'bus'        : 7
        }
        
        # tf bilinear interpolation is different from any other's, just make do
        self.input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
        self.input_size = self.input_sizes[self.compound_coef] if self.force_input_size is None else self.force_input_size

        # load model
        self.model = EfficientDetBackbone(compound_coef=self.compound_coef, num_classes=len(self.obj_list))
        self.model.load_state_dict(torch.load(f'aiannotator/objectdetection/efficientnet/weights/efficientdet-d{self.compound_coef}.pth'))
        self.model.requires_grad_(False)
        self.model.eval()

        if self.use_cuda:
            self.model = self.model.cuda()
        if self.use_float16:
            self.model = self.model.half()

        # Box
        self.regressBoxes = BBoxTransform()
        self.clipBoxes    = ClipBoxes()

    def Detect(self, image):

        if image.shape[2] == 4:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 1:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        ori_imgs, framed_imgs, framed_metas = preprocess_video(rgb_image, max_size=self.input_size)
        
        if self.use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not self.use_float16 else torch.float16).permute(0, 3, 1, 2)

        # model predict
        with torch.no_grad():
            features, regression, classification, anchors = self.model(x)

            out = postprocess(x,
                            anchors, regression, classification,
                            self.regressBoxes, self.clipBoxes,
                            self.threshold, self.iou_threshold)

        # result
        out = invert_affine(framed_metas, out)

        # function for display
        def display(preds, imgs):
            for i in range(len(imgs)):
                if len(preds[i]['rois']) == 0:
                    return imgs[i]

                for j in range(len(preds[i]['rois'])):
                    (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                    cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
                    obj = self.obj_list[preds[i]['class_ids'][j]]
                    score = float(preds[i]['scores'][j])

                    cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                                (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 0), 1)
                
                return imgs[i]

        def output_shapes(preds, imgs):
            """ label, points """
            shapes = []
            for i in range(len(imgs)):
                if len(preds[i]['rois']) == 0:
                    return shapes

                for j in range(len(preds[i]['rois'])):
                    (x_min, y_min, x_max, y_max) = preds[i]['rois'][j].astype(np.int)
                    points = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
                    label = self.obj_list[preds[i]['class_ids'][j]]

                    if label in self.out_class_id_dict.keys():
                        shapes.append((label, points, None, None, True))
                
                return shapes

        # show frame by frame
        shapes = output_shapes(out, ori_imgs)

        return shapes



        