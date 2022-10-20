import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms.functional as F
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.utils import draw_segmentation_masks

from . import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    return model


def load_saved_model(model_name="model/car-damage-mask-rcnn.pth"):
    # Download model from Box into model/ folder
    model_path = Path(model_name)
    if not model_path.is_file():
        model_url = "https://ibm.box.com/shared/static/yxzgn5te17itbrgqdfw31twvxeip5sro.pth"
        print("Model weights not detected. Downloading weights file...")
        with requests.get(model_url, stream=True) as r:
            with open(model_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        print("Download complete.")

    model = get_instance_segmentation_model(num_classes=2)
    model.load_state_dict(torch.load(model_name))
    model.eval()
    return model


def predict(model, img):
    img_t, _ = get_transform(train=False)(img, None)

    with torch.no_grad():
        prediction = model([img_t])

    return prediction


def visualize(img_int, prediction, score_thresh=0.75):

    proba_threshold = 0.5
    max_pred_score = prediction[0]["scores"].max()
    score_threshold = score_thresh * max_pred_score

    boolean_masks = [
        out["masks"][out["scores"] > score_threshold] > proba_threshold
        for out in prediction
    ]

    car_damage_with_masks = [
        draw_segmentation_masks(img, mask.squeeze(1))
        for img, mask in zip([img_int], boolean_masks)
    ]

    def show(imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        for i, img in enumerate(imgs):
            img = img.detach()
            img = F.to_pil_image(img)
        return img

    return np.asarray(show(car_damage_with_masks))
