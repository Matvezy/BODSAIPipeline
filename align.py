from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import math

class Aligner:
  def __init__(self, height):
    self.height = height
    self.ppm_std = 1.56

  def findends(self, mask):
    masklen = len(mask) - 1
    trig1 = False
    trig2 = False
    trig3 = False
    trig4 = False
    while masklen >= 0:
      for j in mask[masklen]:
        if j > 0.5:
          trig1 = True
          break
      if trig1:
        break
      masklen -= 1

    topix = 0
    while topix < len(mask):
      for j in mask[topix]:
        if j > 0.5:
          trig2 = True
          break
      if trig2:
        break 
      topix += 1

    left = 0
    for i in range(len(mask[0])):
      for j in range((len(mask))):
        if mask[j][i] > 0.5:
          trig4 = True
          break
      if trig4:
        break
      left += 1

    right = len(mask[0]) - 1
    while right >= 0:
      for j in range(len(mask)-1):
        if mask[j][right] > 0.5:
          trig3 = True
          break
      if trig3:
        break
      right -= 1
      
    return masklen, topix, left, right

  def align_image(self, imgo):
    image = mpimg.imread(imgo)
    image = Image.fromarray((image* 255).astype(np.uint8))
    mask = image.split()[-1]
    transform = T.Compose([T.ToTensor()])
    mask = transform(mask)
    bottom, top, left, right = self.findends(mask[0])
    req_pixheight = int(self.ppm_std * self.height)
    image=mpimg.imread(imgo)
    image = image[top:bottom, left:right]
    req_width = int((req_pixheight * image.shape[1])/image.shape[0])
    image = Image.fromarray((image* 255).astype(np.uint8))
    image = TF.resize(image, (req_pixheight, req_width))
    alpha = image.split()[-1]
    alpha = np.array(alpha)
    width = len(alpha[0])
    height1 = len(alpha)
    midpoint_w = 0
    dist_h = 0
    if width < 312:
      dist_w = 312 - width
      if dist_w%2 == 0:
        midpoint_w1 = int(dist_w/2)
        midpoint_w2 = int(dist_w/2)
      else:
        midpoint_w1 = math.ceil(dist_w/2)
        midpoint_w2 = int(dist_w/2)
    if height1 < 312:
      dist_h = 312 - height1
    alpha = np.pad(alpha, ((dist_h, 0),(midpoint_w1, midpoint_w2)), 'constant')
    alpha = Image.fromarray(alpha)
    #alpha_image = Image.merge('RGB', (alpha, alpha, alpha))
    return alpha
