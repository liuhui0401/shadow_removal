import os
import torchvision.transforms as transforms
import torch
import numpy as np
from PIL import Image

input_path = './dataset/test/test_A'
old_path = './ckpt/B_100_mask6'
new_path = './pretrained_model/pretrained_model'
old_list = []
old_dict = {}
transform = transforms.Compose([transforms.ToTensor()])

input_list = []
input_dict = {}
cnt = 0
for file_name in os.listdir(input_path):
    file = os.path.join(input_path, file_name)
    img = Image.open(file).resize((256,256))
    img_tensor = transform(img)
    input_list.append(img_tensor)
    input_dict[str(file_name)] = cnt
    cnt += 1

cnt = 0
for file_name in os.listdir(old_path):
    file = os.path.join(old_path, file_name)
    img = Image.open(file).resize((256,256))
    img_tensor = transform(img)
    old_list.append(img_tensor)
    old_dict[str(file_name)] = cnt
    cnt += 1

for file_name in os.listdir(new_path):
    if file_name.endswith('o.png'):
        file = os.path.join(new_path, file_name)
        img = Image.open(file)
        img_tensor = transform(img)
        new_name = str(file_name).replace('-o.png', '.png')
        num = old_dict[str(new_name)]
        output = torch.cat([input_list[num], old_list[num], img_tensor], axis=-1)
        deal = transforms.ToPILImage()
        output = deal(output)
        if not os.path.exists('./contrastive_improve'):
            os.mkdir('./contrastive_improve')
        save_path = os.path.join('./contrastive_improve', new_name)
        output.save(save_path)