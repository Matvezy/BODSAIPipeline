from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
import torch.nn as nn
from torchvision import datasets, models
from torch.autograd import Variable
import joblib
import pickle

class Network18(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.model_name='resnet18'
        self.model = models.resnet18()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

def findppm(mask, height):
  mask = mask.cpu().numpy()[0]
  masklen = len(mask) - 1
  trig1 = False
  trig2 = False
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
  pixheight = masklen - topix
  ppmratio = pixheight/height
  return ppmratio

def getsize(model, frontPPM, frontWidth, sidePPM = 0, sideWidth = 0):
  if (sidePPM != 0) and (sideWidth != 0):
      inp = np.array([frontPPM, frontWidth, sidePPM, sideWidth])
  else:
      inp = np.array([frontPPM, frontWidth])
  inp = inp.reshape(1, -1)
  loaded_model = pickle.load(open(model, 'rb'))
  pred = loaded_model.predict(inp)
  return pred  
    
class ShotSizePredictor:
  def __init__(self, height, image1, image2):
    self.height = height
    self.image1 = image1
    self.image2 = image2
    
  def image_loader(self, image):
    #image = cv2.imread(image)
    #image = Image.fromarray(image)
    image = TF.to_tensor(image)
    image = TF.normalize(image, [0.5], [0.5])
    image = Variable(image, requires_grad=True)
    return image.cuda()

  def run_net(self, net, img):
    network = Network18()
    network.cuda()
    network.load_state_dict(torch.load(net))
    network.eval()
    dots = (network(img[None, ...]).cpu())
    dots = dots.view(-1, 2 ,2)
    dots = dots.detach().numpy()[0]
    dot1 = dots[0][0]
    dot2 = dots[1][0]
    return dots, dot1, dot2
    
  def get_predictions(self):
    with torch.no_grad():
      imgg = np.asarray(self.image1)
      img1 = self.image_loader(self.image1)
      shoulders_front, shoulders_frontDot1, shoulders_frontDot2 = self.run_net('models/shoulders_dots1.pth', img1)
      RA_front, RA_frontDot1, RA_frontDot2 = self.run_net('models/LA_dots1.pth', img1)
      RL_front, RL_frontDot1, RL_frontDot2 = self.run_net('models/RA_dots1.pth', img1)
      LA_front, LA_frontDot1, LA_frontDot2 = self.run_net('models/RL_dots1.pth', img1)
      LL_front, LL_frontDot1, LL_frontDot2 = self.run_net('models/LL_dots1.pth', img1)
      waist_front, waist_frontDot1, waist_frontDot2 = self.run_net('models/wf1_dots1.pth', img1)
      hips_front, hips_frontDot1, hips_frontDot2 = self.run_net('models/hf1_dots1.pth', img1)
      bust_front, bust_frontDot1, bust_frontDot2 = self.run_net('models/bf1_dots1.pth', img1)   
      plt.imshow(self.image1, cmap='gray')
      plt.scatter(shoulders_front[:,0], shoulders_front[:,1], s=8)
      plt.scatter(RA_front[:,0], RA_front[:,1], s=8)
      plt.scatter(LA_front[:,0], LA_front[:,1], s=8)
      plt.scatter(LL_front[:,0], LL_front[:,1], s=8)
      plt.scatter(RL_front[:,0], RL_front[:,1], s=8)
      plt.scatter(waist_front[:,0], waist_front[:,1], s=8)
      plt.scatter(hips_front[:,0], hips_front[:,1], s=8)
      plt.scatter(bust_front[:,0], bust_front[:,1], s=8)
      plt.show()
      img2 = self.image_loader(self.image2)
      SL_side, SL_sideDot1, SL_sideDot2 = self.run_net('models/SL_dots1.pth', img2)
      SS_side, SS_sideDot1, SS_sideDot2 = self.run_net('models/SS_dots.pth', img2)
      waist_side, waist_sideDot1, waist_sideDot2 = self.run_net('models/ws1_dots1.pth', img2)
      hips_side, hips_sideDot1, hips_sideDot2 = self.run_net('models/hs1_dots1.pth', img2)
      bust_side, bust_sideDot1, bust_sideDot2 = self.run_net('models/bs1_dots1.pth', img2)
      plt.imshow(self.image2, cmap='gray')
      plt.scatter(SL_side[:,0], SL_side[:,1], s=8)
      plt.scatter(SS_side[:,0], SS_side[:,1], s=8)
      plt.scatter(waist_side[:,0], waist_side[:,1], s=8)
      plt.scatter(hips_side[:,0], hips_side[:,1], s=8)
      plt.scatter(bust_side[:,0], bust_side[:,1], s=8)
      plt.show()
      fimg = self.image1
      fimg = TF.resize(fimg, (312,312))
      transform = T.Compose([T.ToTensor()])
      fimg = transform(fimg)
      fimg = TF.normalize(fimg, [0.5], [0.5])
      front_ppm = findppm(fimg, self.height)
      simg = self.image2
      simg = TF.resize(simg, (312,312))
      simg = transform(simg)
      simg = TF.normalize(simg, [0.5], [0.5])
      side_ppm = findppm(simg, self.height)
      print(front_ppm)
      print(side_ppm)
      RA = getsize('l2_models/RA_rf.sav', front_ppm, RA_frontDot1 - RA_frontDot2)
      LA = getsize('l2_models/LA_rf.sav', front_ppm, LA_frontDot2 - LA_frontDot1)
      shoulders = getsize('l2_models/shoulders_rf.sav', front_ppm, shoulders_frontDot1 - shoulders_frontDot2)
      RL = getsize('l2_models/RL_rf.sav', front_ppm, RL_frontDot1 - RL_frontDot2, side_ppm, SL_sideDot2 - SL_sideDot1)
      LL = getsize('l2_models/LL_rf.sav', front_ppm, LL_frontDot2 - LL_frontDot1, side_ppm, SL_sideDot2 - SL_sideDot1)
      waist = getsize('l2_models/waist_rf.sav', front_ppm, waist_frontDot1 - waist_frontDot2, side_ppm, waist_sideDot1 - waist_sideDot2)
      bust = getsize('l2_models/bust_rf.sav', front_ppm, bust_frontDot1 - bust_frontDot2, side_ppm, bust_sideDot1 - bust_sideDot2)
      hips = getsize('l2_models/hips_rf.sav', front_ppm, hips_frontDot1 - hips_frontDot2, side_ppm, hips_sideDot1 - hips_sideDot2)
      print("Arm Loop:" + str((LA+RA)/2))
      print("Leg Loop: " + str((RL+LL)/2))
      print("Shoulders: " + str(shoulders))
      print("Waist: " + str(waist))
      print("Bust: " + str(bust))
      print("Hips: " + str(hips))
