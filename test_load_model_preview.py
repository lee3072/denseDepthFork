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


model = torch.load("eval/76model.h5")
model.eval()
cap = cv2.VideoCapture(-1)
while (True):
    ret, frame = cap.read()
    frame = cv2.resize(frame,dsize=(640,480),interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.moveaxis(frame, -1, 0)
    frame = np.expand_dims(frame, axis=0)
    frame = torch.from_numpy(frame).float()
    output = model(torch.autograd.Variable(frame.cuda())) # data is one-image batch of size [1,3,H,W] where 3 - number of color channels
    output_n = DepthNorm(output)
    output = colorize(vutils.make_grid(output_n.data, nrow=4, normalize=False))
    output = np.moveaxis(output, 0, -1)
    cv2.imshow("frame",output)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()