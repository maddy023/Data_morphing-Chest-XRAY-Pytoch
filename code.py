#it wil create all matrix files needed

import torch
import numpy as np
import os

exist = os.path.exists("mul_matrix.npy")

if exist == False:
    mul_matrix_np = np.random.rand(32, 32)
    reverse_np = np.linalg.inv(mul_matrix_np)
    np.save("mul_matrix.npy", mul_matrix_np)
    np.save("reverse_matrix.npy", reverse_np)

mul_matrix_np = np.load("mul_matrix.npy")
mul_matrix = torch.from_numpy(mul_matrix_np).float()

def encrypt_image(image_tensor):
    viewed_tensor = image_tensor
    result_tensor = torch.randn(3, 32, 32)
    for i in range(3):
        result_tensor[i] = viewed_tensor[i].mm(mul_matrix)
    return result_tensor


#it wil throw error if code is executed on CPU,only 
32x32 input images.Any dataset resize to this dim,before feeding.

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from tqdm import tqdm
 
import numpy as np

use_cuda = torch.cuda.is_available()

function = lambda x:encrypt_image(x)

def val(device, net, valloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in valloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predict = outputs.max(1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    # print the accuracy
    print('Accuracy of the network on the val images: %.3f %%' % (
        100 * correct / total))


def train(model, device, criterion, trainloader, optimizer, epochs, epoch_count=None):
    model.train()
    for epoch in range(epochs):
        epoch += 1
        pbar = tqdm(trainloader, total=len(trainloader))
        train_loss_all = .0

        epoch_print = epoch if epoch_count is None else epoch_count
        for batch_id, (inputs, labels) in enumerate(pbar):

            if use_cuda:
                inputs = inputs.to(device)
                labels = labels.to(device)

            inputs = torch.autograd.Variable(inputs)
            labels = torch.autograd.Variable(labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predict = outputs.max(1)
            train_loss_all += loss.data
            train_loss = train_loss_all/(batch_id+1)
            pbar.set_description("poch: {%d} - loss: {%5f} " % (epoch_print, train_loss))
    return

#remove augmentation as ur work,or add smth else but dont remove the resize

def prepare_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        torchvision.transforms.Resize((32,32)),                                  
        transforms.RandomHorizontalFlip(p=0.6),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.RandomVerticalFlip(p=0.6),
 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Lambda(function)
    ])
    transform_test = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.6),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.RandomVerticalFlip(p=0.6),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #transforms.Lambda(function)
    ])
#From pytorch dataset use cifar ,or anything else according to number of images play with #batch_size.Chnage the location of root to ur dataset dir.

    trainset = torchvision.datasets.ImageFolder(root='/content/chest_xray/chest_xray/train',   transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True, num_workers=2)
    valset = torchvision.datasets.ImageFolder(root='/content/chest_xray/test/',   transform=transform_test)
    valloader = torch.utils.data.DataLoader(valset, batch_size=Batch_Size, shuffle=False, num_workers=2)

    return trainloader, valloader


def prepare_morphed_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.6),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.RandomVerticalFlip(p=0.6),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(function)
    ])
    transform_test = transforms.Compose([
        torchvision.transforms.Resize((32,32)),
        transforms.RandomHorizontalFlip(p=0.6),
        torchvision.transforms.RandomRotation(45),
        torchvision.transforms.RandomVerticalFlip(p=0.6),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(function)
    ])
 
    trainset = torchvision.datasets.ImageFolder(root='/content/chest_xray/chest_xray/train',   transform=transform_train)
    valset = torchvision.datasets.ImageFolder(root='/content/chest_xray/test/',  transform=transform_test)
 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=Batch_Size, shuffle=True,  )
    valloader = torch.utils.data.DataLoader(valset, batch_size=Batch_Size, shuffle=False,  )
    return trainloader, valloader

def datareverse(data):
    trans_mat = np.load("mul_matrix.npy")
    reverse_m_np = np.linalg.inv(trans_mat)
    reverse_m = torch.from_numpy(reverse_m_np).float()
    for i in range(3):
        data[i] = data[i].mm(reverse_m)
    return data

#Creating the neural network based on VGG16 arch,for augmentation extra first layer is added
#to read the morphed data.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

cfg_seperated = [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class seperate_VGG16(nn.Module):
    def __init__(self):
        super(seperate_VGG16, self).__init__()
        self.CM_1 = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_2 = nn.Sequential(*[nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_3 = nn.Sequential(*[nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_4 = nn.Sequential(*[nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.CM_5 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
                                    nn.MaxPool2d(2, 2)])
        self.classifier = nn.Linear(512,10)


    def forward(self, x):
        out = F.pad(x,(1,0,1,0))
        out = self.CM_1(out)
        out = self.CM_2(out)
        out = self.CM_3(out)
        out = self.CM_4(out)
        out = self.CM_5(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


class LC(nn.Module):
    def __init__(self):
        super(LC,self).__init__()
        self.C = nn.Sequential(*[nn.Conv2d(3, 64, kernel_size=3, padding=1)])
        self.L = nn.Linear(3072, 3072, bias=False)

    def forward(self, x):
        out = x.view(Batch_Size, 1, 3072)
        out = self.L(out)
        out = out.view(Batch_Size, 64, 32, 32)
        out = self.C(out)
        return out

class LCVGG16(nn.Module):
    def __init__(self):
        super(LCVGG16, self).__init__()
        self.comb = np.load("Combination.npy").astype(np.float32)
        self.LC = torch.from_numpy(self.comb).cuda()
        self.rest = rest()
        #for param in self.LC.parameters():
            #param.requires_grad = False

    def forward(self, x):
        #out = x.view(Batch_Size, 3*32*32)
        out = x.view(x.size(0), -1)
        out = out.mm(self.LC)
        #out = out.view(Batch_Size, 64, 32, 32)
        out = out.view(out.size(0), 64, 32, 32)
        out = self.rest(out)
        return out


class rest(nn.Module):
    def __init__(self):
        super(rest, self).__init__()
        self.restconv = self._make_layers()
        self.classifier = nn.Linear(512, 10)
       

    def forward(self, x):
        out = self.restconv(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):
        layers = []
        in_channels = 64
        for x in cfg_seperated:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#training the model

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
 

sys.path.append(os.getcwd())

N_Epoch=100
LR = 0.01
if __name__ == '__main__':

    # load data
    trainloader, valloader = prepare_data()

    net = seperate_VGG16()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: ', device)
    net.to(device)
    print('start to train.')
    lr = LR
    for epoch in range(N_Epoch):
        if epoch%2 == 0:
            #lr *= 0.5
            print('start to val.')
            val(device, net, valloader)
            print('save model')
            torch.save(net, ('pretrained_VGG16.pkl'))
        optimizer = optim.SGD(params=net.parameters(), lr=lr, momentum=0.9, weight_decay=2e-3)
        train(net, device, nn.CrossEntropyLoss(), trainloader, optimizer, 1, epoch_count=epoch+1)

    print('start to val.')
    val(device, net, valloader)

    print('save model')
    torch.save(net, 'pretrained_VGG16.pkl')

#creating the combination file,contain the morphed data which further loaded for aug-cnn arch #netowrk.

import torch
import numpy as np
import time

net = torch.load("vgg16_weights_acc_62.pkl")
C = net.CM_1
C_1 = C[0]
related_pixel = [((0,0),-33), ((0,1),-32), ((0,2),-31), ((1,0),-1), ((1,1),0), ((1,2),1), ((2,0),31), ((2,1),32), ((2,2),33)]

reverse_matrix = np.load("reverse_matrix.npy").astype(np.float32)
weight_1 = C_1.weight.detach().cpu().numpy()
bias_1 = C_1.bias.detach().cpu().numpy()


Inverse = np.zeros((3072, 3072), dtype=np.float32)
Comb_1 = np.zeros((3072, 64*32*32), dtype=np.float32)


start_time = time.time()

for i in range(96):
    for j in range(32):
        Inverse[i*32+j][i*32:i*32+32] = reverse_matrix[j]

print("Inverse matrix complete! Time:", time.time()-start_time)


for m in range(0, 64):
    for n in range(0, 1024):
        line_n = n%32
        for idx, l in related_pixel:
            if (n+l) >= 0 and (n+l) < 1024 and (line_n+idx[1]>0) and (line_n+idx[1]<33):
                Comb_1[n+l][m*1024+n] = weight_1[m][0][idx]
                Comb_1[n+l+1024][m*1024+n] = weight_1[m][1][idx]
                Comb_1[n+l+2048][m*1024+n] = weight_1[m][2][idx]

print("Conv_1 complete! Time:", time.time()-start_time)


Inverse = torch.from_numpy(Inverse)
Inverse = Inverse.cuda()

Comb_1 = torch.from_numpy(Comb_1)
Comb_1 = Comb_1.cuda()

Comb_1 = torch.mm(Inverse, Comb_1)

print("Comb_1 complete! Time:", time.time()-start_time)

Comb_1 = Comb_1.cpu().numpy()
Comb_1 = Comb_1.astype(np.float16)
np.save("Combination.npy", Comb_1)
np.save("Bias.npy", bias_1)

print("Comb_1 Saved:", time.time()-start_time)

import os
import torch
import torch.nn as nn
import torch.optim as optim
import sys
 

sys.path.append(os.getcwd())

N_Epoch = 100
Batch_Size=50
LR = 0.05

if __name__ == '__main__':

    # load data
    trainloader, valloader = prepare_morphed_data()

    net = LCVGG16()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: ', device)
    net.to(device)
    print('start to train.')
    lr = LR
    for epoch in range(N_Epoch):
        if epoch%2 == 0:
            #lr *= 0.5
            print('start to val.')
            val(device, net, valloader)
            print('save model')
            torch.save(net, 'Aug_conv_VGG16.pkl')
        optimizer = optim.SGD(params=net.rest.parameters(), lr=lr, momentum=0.9, weight_decay=2e-3)
        train(net, device, nn.CrossEntropyLoss(), trainloader, optimizer, 1, epoch_count=epoch+1)

    print('start to val.')
    val(device, net, valloader)

    torch.save(net, 'Aug_conv_VGG16.pkl')
    print('save model')
