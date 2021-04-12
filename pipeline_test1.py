from align import Aligner
from predictor import ShotSizePredictor
from PIL import Image
import cv2

height = 179

# Aligning front and side images
aligner = Aligner(height)
aligned_front = aligner.align_image('assets/Front.png')     
aligned_side = aligner.align_image('assets/Side.png') 
# Getting model preditcions dictionary
predictor = ShotSizePredictor(height, aligned_front, aligned_side)
predictor.get_predictions()


