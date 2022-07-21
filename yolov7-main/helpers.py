from from pathlib import Path

from yolov7-main.models.common import AutoShape, DetectMultiBackend
from yolov7-main.utils.general import LOGGER, logging
from yolov7-main.utils.torch_utils import torch


def load_model(model_path, device=None, autoshape=True, verbose=False):
    """
    Creates a specified YOLOv5 model
    Arguments:
        model_path (str): path of the model
        device (str): select device that model will be loaded (cpu, cuda)
        pretrained (bool): load pretrained weights into the model
        autoshape (bool): make model ready for inference
        verbose (bool): if False, yolov5 logs will be silent
    Returns:
        pytorch model
    (Adapted from yolov5.hubconf.create)
    """
    # set logging
    if not verbose:
        LOGGER.setLevel(logging.WARNING)

    # set device if not given
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif type(device) is str:
        device = torch.device(device)

    model = DetectMultiBackend(model_path, device=device)

    if autoshape:
        model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS
    return model.to(device)
