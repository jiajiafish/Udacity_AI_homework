#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import time
from matplotlib import cm
from PIL import Image
import argparse

supported_arch = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet169',
    'densenet161',
    'densenet201'
]
parser = argparse.ArgumentParser(description="训练一个图像识别神经网络",
                      usage="python ./train.py flowers/ --gpu --learning_rate 0.001 --hidden_units1 3136 --hidden_units2 784--epochs 5",
                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
            "data_dir",
            action="store",
            type=str,
            help ="图片文件存放位置")

parser.add_argument(
            '--categories_json',
             action="store",
             default="cat_to_name.json",
             dest='categories_json',
             type=str,
             help='Path to file containing the categories.',
                        )
parser.add_argument(
            "--arch", 
            action="store",
            default="vgg16",
            type=str,
            help='支持的框架' + ", ".join(supported_arch))

parser.add_argument(
            '--gpu',
            action="store_true",
            default=False,
            help='Use GPU')

parser.add_argument('--save_name',
                        action="store",
                        default="checkpoint.pth",
                        type=str,
                        help='Checkpoint 文件名.',
                        )
# 超参数包含训练速度，epochs数目，隐藏层的数量
hyper_parameters = parser.add_argument_group('hyperparameters')

hyper_parameters.add_argument(
    "--hidden_units1",
    action = "store",
    default=3136,
    type = int,
    help="隐藏层1单元数")

hyper_parameters.add_argument(
    "--hidden_units2",
    action = "store",
    default=784,
    type = int,
    help="隐藏层1单元数")

hyper_parameters.add_argument(
    "--epochs",
    action ="store",
    default=3,
    type = int,
    help="epochs数")

hyper_parameters.add_argument(
    "--learning_rate", 
    action="store",
    default = 0.001,
    type =float,
    help="学习速率")





args = parser.parse_args()
print("用来训练的目标文件夹在：", args.data_dir)
print("categories_json是：",args.categories_json)
print("选择的架构是：",args.arch)
print("保存的文件名是：",args.save_name)
print("第一隐藏层单元数：",args.hidden_units1)
print("第二隐藏层单元数：",args.hidden_units2)
print("epochs数目：",args.epochs)
print("学习速度是：",args.learning_rate)




# python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20 --arch "vgg13"  --save_dir save_directory --gpu
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("当前计算机不支持GPU模式，将会继续采用CPU模式")
        device = torch.device("cpu")
else:
    device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_dir = 'train'
valid_dir = 'valid'
test_dir = 'test'

# data_dir='flowers/'
data_dir = args.data_dir

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
# TODO: Load the datasets with ImageFolder

train_data = datasets.ImageFolder(data_dir + train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(data_dir + valid_dir, transform=valid_transforms)
test_data = datasets.ImageFolder(data_dir + test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)

testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


# ### 标签映射

import json

with open(args.categories_json, 'r') as f:
    cat_to_name = json.load(f)
print(len(cat_to_name))

if not args.arch.startswith("vgg") and not args.arch.startswith("densenet"):
    print("只支持 VGG 和 DenseNet神经网络")
    exit(1)
# model = models.vgg16(pretrained=True)
model = models.__dict__[args.arch](pretrained=True)
densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

# 根据要求选择vgg16作为迁移训练的模型。
if args.arch.startswith("vgg"):
    input_size = model.classifier[0].in_features
else:
    input_size = densenet_input[args.arch]
# 我有两个hidden layer
# hidden_size1 = input_size // 8
hidden_size1 = args.hidden_units1

# hidden_size2 = input_size // 32
hidden_size2 = args.hidden_units2

output_size = len(cat_to_name)
# ## 测试网络
for param in model.parameters():
    param.requires_grad = False
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
#     根据vgg16的模型参数来设定自己的模型分类器
        ('fc1', nn.Linear(input_size, hidden_size1)),
        ('relu1', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_size1, hidden_size2)),
        ('relu2', nn.ReLU()),
        ('dropout', nn.Dropout(p=0.5)),
        ('output', nn.Linear(hidden_size2, output_size)),
        ('softmax', nn.LogSoftmax(dim=1))
#     softmax是用来解决多分类问题的
    ]))
model.classifier = classifier
for i in model.classifier:
    print(i)
model.classifier[0].in_features
len(validloader.batch_sampler)
len(trainloader.batch_sampler)
dir(trainloader)
epochs = args.epochs
print_every = 50
lr=args.learning_rate
model.zero_grad()

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

# Move model to perferred device.
model = model.to(device)


data_set_len = len(trainloader.batch_sampler)
total_val_images = len(validloader.batch_sampler) * validloader.batch_size

print(f'正在使用 {device} 训练')
print(f'批量数是 {data_set_len} 每一批抽样数为 {trainloader.batch_size}.')

for e in range(epochs):
    e_loss = 0
    prev_chk = 0
    total = 0
    correct = 0
    print(f'\n 第 {e+1}个 Epoch 共有{epochs}个 epoch\n----------------------------')
    
    for ii, (images, labels) in enumerate(trainloader):
#         print(ii)
        
        images,labels = images.to(device),labels.to(device)
#         print("images")
#         print(images)
#         print("labels")
#         print(labels)
        
        # Set gradients of all parameters to zero. 
        optimizer.zero_grad()
        
        # Propigate forward and backward 
        #参考迁移学习代码
        outputs = model.forward(images)
#         print(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #running_loss += loss.item()
        e_loss += loss.item()
        
        # Accuracy
        _, predicted = torch.max(outputs.data, 1)
#         print(torch.max(outputs.data, 1))
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Keep a running total of loss for
        # this epoch
        itr = (ii + 1)
        if itr % print_every == 0:
            avg_loss = f'avg. loss: {e_loss/itr:.4f}'
            acc = f'accuracy: {(correct/total) * 100:.2f}%'
            print(f'  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
            prev_chk = (ii + 1)
    # Validate Epoch
    valid_correct = 0
    valid_total = 0
    

    # Disabling gradient calculation
    with torch.no_grad():
        model.eval()
        for ii, (images, labels) in enumerate(validloader):
            # Move images and labeles perferred device
            # if they are not already there
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
#             torch.max会返回两个值,这种写法是python的固定写法,废弃第一个值
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
        print(f"\n\tValidating for epoch {e+1}...")
        correct_perc = 0
        if valid_correct > 0:
            correct_perc = (100 * valid_correct / valid_total)
        print(f'\tAccurately classified {correct_perc}% of {total_val_images} images.')
        model.train()

print('完成...正在保存到目录中去')
def save_checkpoint(model_state, file='default_checkpoint.pth'):
    torch.save(model_state, file)

# 属性附加到模型上，这样稍后推理会更轻松。
model.class_to_idx = train_data.class_to_idx
# 保存相关参数
model_state = {
    'arch':args.arch,
    'epoch': epochs,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'classifier': classifier,
    'class_to_idx': model.class_to_idx,
}

save_checkpoint(model_state, args.save_name)
print("神经网络参数已经被记录到"+args.save_name)
