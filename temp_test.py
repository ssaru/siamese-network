import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.utils
import imgaug as ia
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
from imgaug import augmenters as iaa
from PIL import Image
from torchsummaryX import summary

from siamese_network_defect import SiameseNetwork, DefectDataset, imshow, Augmenter, Config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = SiameseNetwork().to(device)
net.summary()
exit()
net.load_state_dict(torch.load("./result.pth.tar", map_location=device)["state_dict"])
net.eval()
seq = iaa.Sequential([
            #iaa.Resize({"height": 100, "width": 100})
            ])

composed = transforms.Compose([Augmenter(seq)])

dataset = DefectDataset(root=Config.testing_dir, transform=composed)

vis_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=0,
                        batch_size=8)

dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
print(example_batch[0].shape)


test_dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)

for i in range(10):
    x0, x1, label2 = next(dataiter)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = net(Variable(x0).to(device), Variable(x1).to(device))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow(torchvision.utils.make_grid(concatenated), 'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
    if label2 == 0:
        print("same")
    elif label2 == 1:
        print("differ")
