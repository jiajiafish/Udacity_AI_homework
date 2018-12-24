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

parser = argparse.ArgumentParser(
        description="Image prediction.",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
parser.add_argument('path_to_image',
                        help='Path to image file.',
                        default = "./flowers/valid/102/image_08041.jpg",
                        action="store")

parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        default = "checkpoint.pth",
                        action="store")


parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        type=int,
                        help='Return top KK most likely classes.',
                     )

parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        type=str,
                        help='Path to file containing the categories.',
                     )

parser.add_argument('--gpu',
                     action="store_true",
                     default=False,
                     help='Use GPU')

args = parser.parse_args()


print("选择的文件路径是：",args.path_to_image)
print("模型的载入文件是：",args.checkpoint_file)
print("高概率的个数：",args.top_k)
print("训练模式选择GPU：",args.gpu)
print("categories_json是：",args.category_names)
def load_checkpoint(file='default_checkpoint.pth'):
    model_state = torch.load(file, map_location=lambda storage, loc: storage)
    
#     根据上面设定的参数load
    model = models.__dict__[model_state['arch']](pretrained=True)
    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]
    #PIL引入图片
    pil_image = Image.open(image).convert("RGB")
    
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image

chk_image = process_image('./flowers/valid/1/image_06739.jpg')
torch.no_grad()
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    
    model.eval()
    image = process_image(image_path)
    # cpu mode
    if args.gpu:
        if torch.cuda.is_available():
            model.cuda()
            image.cuda()
        else:
            model.cpu()
            print("当前系统不支持cuda，将采用cpu的形式")
    else:  
        model.cpu()
    
    # load image as torch.Tensor
#     print(image)
    # Unsqueeze returns a new tensor with a dimension of size one
    # https://blog.csdn.net/flysky_jay/article/details/81607289
    image = image.unsqueeze(0)
    
    # Disabling gradient calculation 
    # (not needed with evaluation mode?)
    with torch.no_grad():
    #把图像带入到神经网络
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)
        
        # Calculate the exponentials
        top_prob = top_prob.exp()
#         print(top_prob)
#         print(top_labels)
      
        #成字典
    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()
    
    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])
        
    return top_prob.numpy()[0], mapped_classes

chkp_model = load_checkpoint(args.checkpoint_file)
chk_img_file = args.path_to_image
top_prob, top_classes = predict(chk_img_file, chkp_model,topk=args.top_k)
print(top_prob)
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
top_name = [cat_to_name[x] for x in top_classes]
print(top_name)
top_prob = [x*100for x in top_prob]
print("打印出概览最高的种类。")
print("----------------------------------------")
for i in range(len(top_prob)):
    print("{:n}  {:<15s}{:>15.5f}%".format(i,top_name[i],top_prob[i]))