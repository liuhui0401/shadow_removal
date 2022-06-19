from model import Generator_S2F,Generator_F2S
import os
from os.path import exists, join as join_paths
import cv2
import torch
import pdb
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import random
import copy

netG_A2B = Generator_S2F().cuda()
netG_1 = Generator_S2F().cuda()
netG_A2B_path = 'ckpt/ckpt/netG_A2B.pth'
netG_1_path = 'ckpt/ckpt/netG_1.pth'
netG_A2B.load_state_dict(torch.load(netG_A2B_path))
netG_1.load_state_dict(torch.load(netG_1_path))
netG_A2B.eval()
trainA_path = 'dataset/train_A' # pseudo shadow
trainB_path = 'dataset/train_B'
trainC_path = 'dataset/train_C'

with torch.no_grad():
    for filename in os.listdir(trainA_path):
        A_path = os.path.join(trainA_path, filename)
        i = random.randint(0, 48)
        j = random.randint(0, 48)
        k = random.randint(0,100)
        A_img = color.rgb2lab(io.imread(A_path))
        A_img = resize(A_img,(448,448,3))
        A_img = A_img[i:i+400,j:j+400,:]
        AA_img = copy.deepcopy(A_img)
        A_img = resize(A_img, (480,640,3))
        A_img = color.lab2rgb(A_img)
        if k > 50:
            A_img = np.fliplr(A_img)
        AA_img[:,:,0] = np.asarray(AA_img[:,:,0])/50.0-1.0
        AA_img[:,:,1:] = 2.0*(np.asarray(AA_img[:,:,1:])+128.0)/255.0-1.0
        AA_img = torch.from_numpy(AA_img.copy()).float().cuda()
        AA_img = AA_img.view(400,400,3)
        AA_img = AA_img.transpose(0, 1).transpose(0, 2).contiguous().unsqueeze(0)

        pesudo = netG_A2B(AA_img)
        off_pesudo = netG_1(pesudo)

        pesudo = pesudo.data
        pesudo[:,0] = 50.0*(pesudo[:,0]+1.0)
        pesudo[:,1:] = 255.0*(pesudo[:,1:]+1.0)/2.0-128.0
        pesudo = pesudo.squeeze(0).cpu()
        pesudo = pesudo.transpose(0,2).transpose(0,1).contiguous().numpy()
        pesudo = resize(pesudo, (480,640,3))
        pesudo = color.lab2rgb(pesudo)
        temp_save_path = join_paths('./dataset/test/test_E'+'/%s'%(filename))
        io.imsave(temp_save_path, pesudo)

        off_pesudo = off_pesudo.data
        off_pesudo[:,0] = 50.0*(off_pesudo[:,0]+1.0)
        off_pesudo[:,1:] = 255.0*(off_pesudo[:,1:]+1.0)/2.0-128.0
        off_pesudo = off_pesudo.squeeze(0).cpu()
        off_pesudo = off_pesudo.transpose(0,2).transpose(0,1).contiguous().numpy()
        off_pesudo = resize(off_pesudo, (480,640,3))
        off_pesudo = color.lab2rgb(off_pesudo)
        temp_save_path = join_paths('./dataset/test/test_F'+'/%s'%(filename))
        io.imsave(temp_save_path, off_pesudo)

        C_path = os.path.join(trainC_path, filename)
        C_img = color.rgb2lab(io.imread(C_path))
        C_img = resize(C_img,(448,448,3))
        C_img = C_img[i:i+400,j:j+400,:]
        C_img = resize(C_img, (480,640,3))
        if k > 50:
            C_img = np.fliplr(C_img)
        C_img = color.lab2rgb(C_img)
        testA_img = C_img - A_img + pesudo
        temp_save_path = join_paths('./dataset/test/test_A'+'/%s'%(filename))
        io.imsave(temp_save_path, testA_img)

        B_path = os.path.join(trainB_path, filename)
        B_img = color.rgb2lab(io.imread(B_path))
        B_img = resize(B_img,(448,448,3))
        B_img = B_img[i:i+400,j:j+400,:]
        B_img = resize(B_img, (480,640,3))
        if k > 50:
            B_img = np.fliplr(B_img)
        B_img = color.lab2rgb(B_img)
        B_img = color.rgb2gray(B_img)
        temp_save_path = join_paths('./dataset/test/test_B'+'/%s'%(filename))
        io.imsave(temp_save_path, B_img)

        testC_img = C_img - A_img + off_pesudo
        temp_save_path = join_paths('./dataset/test/test_C'+'/%s'%(filename))
        io.imsave(temp_save_path, testC_img)