import torch
from torch.utils.data import IterableDataset
import queue
import torchvision.transforms.functional as TF
from PIL import Image
import cv2
import numpy as np
import torchvision.utils as vutils
from utils import DepthNorm, colorize
from data import getTrainingTestingData


model = torch.load("eval_new_half/125model.h5")
model.eval()
cap = cv2.VideoCapture(-1)
while (True):
    ret, frame = cap.read()
    loss_scale = 100
    depth_scale = 0.0002500000118743628 * loss_scale
    frame = cv2.resize(frame,dsize=(640,480),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.moveaxis(frame, -1, 0)
    frame = np.expand_dims(frame, axis=0)
    frame = torch.from_numpy(frame/loss_scale).float()
    output = model(torch.autograd.Variable(frame.cuda())) # data is one-image batch of size [1,3,H,W] where 3 - number of color channels
    output2 = output.detach().cpu().numpy()[0][0] * depth_scale
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output2, alpha=255/output2.max()), cv2.COLORMAP_JET)
    cv2.imshow("frame",depth_colormap)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
