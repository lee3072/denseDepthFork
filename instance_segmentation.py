
import torch
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms.functional as F
#####################################
# .. _instance_seg_output:
#
# Instance segmentation models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Instance segmentation models have a significantly different output from the
# semantic segmentation models. We will see here how to plot the masks for such
# models. Let's start by analyzing the output of a Mask-RCNN model. Note that
# these models don't require the images to be normalized, so we don't need to
# use the normalized batch.
#
# .. note::
#
#     We will here describe the output of a Mask-RCNN model. The models in
#     :ref:`object_det_inst_seg_pers_keypoint_det` all have a similar output
#     format, but some of them may have extra info like keypoints for
#     :func:`~torchvision.models.detection.keypointrcnn_resnet50_fpn`, and some
#     of them may not have masks, like
#     :func:`~torchvision.models.detection.fasterrcnn_resnet50_fpn`.

def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

from torchvision.io import read_image
from pathlib import Path
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection import maskrcnn_resnet50_fpn
model = maskrcnn_resnet50_fpn(pretrained=True, progress=False)
model = model.eval()

dog1_int = read_image(str(Path('data/img') / '000154_in.jpg'))
batch_int = torch.tensor(dog1_int).unsqueeze_(0)
# batch_int = torch.stack([dog1_int, dog2_int])
batch = convert_image_dtype(batch_int, dtype=torch.float)
output = model(batch)
print(output)

#####################################
# Let's break this down. For each image in the batch, the model outputs some
# detections (or instances). The number of detections varies for each input
# image. Each instance is described by its bounding box, its label, its score
# and its mask.
#
# The way the output is organized is as follows: the output is a list of length
# ``batch_size``. Each entry in the list corresponds to an input image, and it
# is a dict with keys 'boxes', 'labels', 'scores', and 'masks'. Each value
# associated to those keys has ``num_instances`` elements in it.  In our case
# above there are 3 instances detected in the first image, and 2 instances in
# the second one.
#
# The boxes can be plotted with :func:`~torchvision.utils.draw_bounding_boxes`
# as above, but here we're more interested in the masks. These masks are quite
# different from the masks that we saw above for the semantic segmentation
# models.

dog1_output = output[0]
dog1_masks = dog1_output['masks']
print(f"shape = {dog1_masks.shape}, dtype = {dog1_masks.dtype}, "
      f"min = {dog1_masks.min()}, max = {dog1_masks.max()}")

#####################################
# Here the masks corresponds to probabilities indicating, for each pixel, how
# likely it is to belong to the predicted label of that instance. Those
# predicted labels correspond to the 'labels' element in the same output dict.
# Let's see which labels were predicted for the instances of the first image.

inst_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

inst_class_to_idx = {cls: idx for (idx, cls) in enumerate(inst_classes)}

print("For the first dog, the following instances were detected:")
print([inst_classes[label] for label in dog1_output['labels']])

#####################################
# Interestingly, the model detects two persons in the image. Let's go ahead and
# plot those masks. Since :func:`~torchvision.utils.draw_segmentation_masks`
# expects boolean masks, we need to convert those probabilities into boolean
# values. Remember that the semantic of those masks is "How likely is this pixel
# to belong to the predicted class?". As a result, a natural way of converting
# those masks into boolean values is to threshold them with the 0.5 probability
# (one could also choose a different threshold).

proba_threshold = 0.5
dog1_bool_masks = dog1_output['masks'] > proba_threshold
print(f"shape = {dog1_bool_masks.shape}, dtype = {dog1_bool_masks.dtype}")

# There's an extra dimension (1) to the masks. We need to remove it
dog1_bool_masks = dog1_bool_masks.squeeze(1)

show(draw_segmentation_masks(dog1_int, dog1_bool_masks, alpha=0.9))

#####################################
# The model seems to have properly detected the dog, but it also confused trees
# with people. Looking more closely at the scores will help us plotting more
# relevant masks:

print(dog1_output['scores'])

#####################################
# Clearly the model is less confident about the dog detection than it is about
# the people detections. That's good news. When plotting the masks, we can ask
# for only those that have a good score. Let's use a score threshold of .75
# here, and also plot the masks of the second dog.

score_threshold = .75

boolean_masks = [
    out['masks'][out['scores'] > score_threshold] > proba_threshold
    for out in output
]

dogs_with_masks = [
    draw_segmentation_masks(img, mask.squeeze(1))
    for img, mask in zip(batch_int, boolean_masks)
]
show(dogs_with_masks)
plt.show()
#####################################
# The two 'people' masks in the first image where not selected because they have
# a lower score than the score threshold. Similarly in the second image, the
# instance with class 15 (which corresponds to 'bench') was not selected.