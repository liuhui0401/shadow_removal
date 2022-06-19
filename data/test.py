import os
from PIL import Image
import cv2

path = '../dataset/train/train_A'
for file in os.listdir(path):
    img_path = os.path.join(path, file)
    #A_img = Image.open(img_path).convert('RGB')
    # b, c, h, w = A_img.shape
    # print(b)
    img = cv2.imread(img_path)
    print(img.shape)
    break