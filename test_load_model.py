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

# class MyDataset(IterableDataset):
#     def __init__(self, image_queue):
#       self.queue = image_queue

#     def read_next_image(self):
#         while self.queue.qsize() > 0:
#             # you can add transform here
#             yield self.queue.get()
#         return None

#     def __iter__(self):
#         return self.read_next_image()

model1 = torch.load("eval/149model.h5")
model1.eval()
model2 = torch.load("eval/249model.h5")
model2.eval()
# image_path = "test_load_model_input.png"
# buffer = queue.Queue()
# new_input = Image.open(image_path)
# buffer.put(TF.to_tensor(new_input)) 
# dataset = MyDataset(buffer)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
train_loader, test_loader = getTrainingTestingData(batch_size=1,folder="data_test")
loss_scale = 100
depth_scale = 0.0002500000118743628 * loss_scale
for i, sample_batched in enumerate(test_loader):
    # Prepare sample and target
    image = torch.autograd.Variable(sample_batched['image'].cuda())
    depth = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

    output1 = model1(image.cuda()).detach().cpu().numpy()[0][0] * depth_scale # data is one-image batch of size [1,3,H,W] where 3 - number of color channels
    output2 = model2(image.cuda()).detach().cpu().numpy()[0][0] * depth_scale # data is one-image batch of size [1,3,H,W] where 3 - number of color channels
    depth = depth.detach().cpu().numpy()[0][0] * depth_scale

    depth_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(output1, alpha=255/output1.max()), cv2.COLORMAP_JET)
    diff_colormap1 = cv2.applyColorMap(cv2.convertScaleAbs(output1-depth, alpha=255/(output1-depth).max()), cv2.COLORMAP_JET)
    depth_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(output2, alpha=255/output2.max()), cv2.COLORMAP_JET)
    diff_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(output2-depth, alpha=255/(output2-depth).max()), cv2.COLORMAP_JET)
    truth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255/depth.max()), cv2.COLORMAP_JET)
    diff_diff_colormap = cv2.applyColorMap(cv2.convertScaleAbs(output2-output1, alpha=255/(output2-output1).max()), cv2.COLORMAP_JET)
    cv2.imshow("frame_149",depth_colormap1)
    cv2.imshow("diff_149",diff_colormap1)
    cv2.imshow("truth",truth_colormap)
    cv2.imshow("frame_249",depth_colormap2)
    cv2.imshow("diff_249",diff_colormap2)
    cv2.imshow("diff_149&249",diff_diff_colormap)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
