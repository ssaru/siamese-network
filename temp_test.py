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

# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = SiameseNetwork(size=(250, 250))
net.summary()

net.cnn1.register_forward_hook(get_activation('conv1'))

if device.type == 'cpu':
    model = torch.nn.DataParallel(net)
else:
    model = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()


model.to(device)

model.load_state_dict(torch.load("./result.pth.tar", map_location=device)["state_dict"])
model.eval()

seq = iaa.Sequential([
            iaa.Resize({"height": Config.RESIZE[0], "width": Config.RESIZE[1]})
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

for j in range(2):
    dataiter = iter(test_dataloader)
    for i in range(len(dataset)):
        x0, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = model(Variable(x0).to(device), Variable(x1).to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)
        distance = euclidean_distance.item()

        imshow(torchvision.utils.make_grid(concatenated), 'Pred : {}, Label : {}, Dissimilarity: {:.2f}'
                                                            .format("Same" if distance < 1.5 else "Differ",
                                                                    "Same" if label2 == 0 else "Differ",
                                                                    euclidean_distance.item()),
               should_save=True, name=str(j)+str(i))

        act = activation['conv1'].squeeze()
        for idx in range(act.size(0)):
            plt.figure()
            plt.imshow(act[idx], cmap='gray')
            plt.savefig(str(j)+str(i)+"_"+str(idx)+"_activation")