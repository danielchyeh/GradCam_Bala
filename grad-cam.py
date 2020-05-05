import torch
from torch.autograd import Variable
from torch.autograd import Function
#from torchvision import models
from torchvision import utils
import cv2
import sys
import numpy as np
import argparse
import torch.backends.cudnn as cudnn

import models
import datasets
import math

import os


import scipy
import scipy.misc
from skimage.filters import gaussian as gaussian_filter

from apex.fp16_utils import FP16_Optimizer

#model = models.__dict__['resnet50'](pretrained=True)
#resnet = models.resnet18(pretrained=True)
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        count = 0
        co = 0
        for name, module in self.model.module._modules.items():
            if count < 9:
                x = module(x)
                print('name=',name)
                print('x.size()=',x.size())
                #print(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
                    co+=1
                count+=1
        print(co)
        
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output  = self.feature_extractor(x)
        print(output.size())
        output = output.view(output.size(0), -1)
        print('classfier=',output.size())
        #print(target_activations)
        #exit(0)
        output = self.model.module.fc_new(output)
        #print(output.size())
        return target_activations, output

def preprocess_image(img):
    means=[0.485, 0.456, 0.406]
    stds=[0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        
        
    #Bala algorithm of data argumentation
    #img = np.array(imgs[index:index+1,:,:,:])[0]
    #img = scipy.misc.imresize(preprocessed_img, (224, 224))
            #print('img shape: {}'.format(img.shape))

    og_img = preprocessed_img.copy() #skimage.color.rgb2lab(img)[:,:,0]

    sigma = 9
            
    img = og_img - gaussian_filter(og_img, sigma=sigma, multichannel=True)
    #img = img.transpose((2, 0, 1))
    preprocessed_img =img #(img/255.) - .5
    
    
    
    
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask, save_path, img_name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path+'/'+img_name, np.uint8(255 * cam))

class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input) 

    def __call__(self, input, index = None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input), torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output, positive_mask_1), positive_mask_2)

        return grad_input

class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index = None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0,:,:,:]

        return output

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/naicha.jpg',
                        help='Input image path')
    
    parser.add_argument('--lr', default=0.03, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
    parser.add_argument('--test-only', action='store_true', help='test only')
    parser.add_argument('--low-dim', default=128, type=int,
                    metavar='D', help='feature dimension')
    parser.add_argument('--save-path', type=str, default='./examples/both.png',
                        help='save image path')
    parser.add_argument('--static_loss', default=25, type=float, help='set static loss for apex optimizer')
    
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
   
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()

    
    from os import walk

    f = []
    for (dirpath, dirnames, filenames) in walk(args.image_path):
        f.extend(filenames)
        #print(filenames)
    #print(f)
    
    for ids, img_name in enumerate(f):
    
        model = models.__dict__['resnet50'](low_dim=args.low_dim)
        model = torch.nn.DataParallel(model).cuda()


        optimizer = torch.optim.SGD(model.parameters(), 0.03,
                                    momentum=0.9,
                                    weight_decay=1e-4)    
        # optionally resume from a checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.static_loss, verbose=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                lemniscate = checkpoint['lemniscate']
                print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

        cudnn.benchmark = True


        # Can work with any model, but it assumes that the model has a 
        # feature method, and a classifier method,
        # as in the VGG models in torchvision.


        #model = models.resnet50(pretrained=True)
        #del model.module.fc
        #print(model)
        #print(net.module._modules.items())
        #exit(0)

        #print(model)
        grad_cam = GradCam(model, \
                        target_layer_names = ["layer4"], use_cuda=args.use_cuda)

    #     print(grad_cam)
    #     print(type(grad_cam))
    #     exit(0)

        img = cv2.imread(args.image_path+'/'+img_name, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        print('input.size()=',input.size())
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested index.
        target_index =None

        mask = grad_cam(input, target_index)
        #print(type(mask))

        show_cam_on_image(img, mask, args.save_path, img_name)

        gb_model = GuidedBackpropReLUModel(model = model, use_cuda=args.use_cuda)
        gb = gb_model(input, index=target_index)
        #utils.save_image(torch.from_numpy(gb), 'gb.jpg')

        cam_mask = np.zeros(gb.shape)
        for i in range(0, gb.shape[0]):
            cam_mask[i, :, :] = mask

        cam_gb = np.multiply(cam_mask, gb)
        #utils.save_image(torch.from_numpy(cam_gb), 'cam_gb.jpg')