import sys
import torch
from torchvision import models
from config import opt


def get_densenet201():
    num_cls = len(opt.attributes)
    model = models.densenet201(weights=opt.weights)
    model.classifier = torch.nn.Sequential(torch.nn.Linear(model.classifier.in_features, num_cls), torch.nn.Sigmoid())
    return model


def get_resnet152():
    num_cls = len(opt.attributes)
    model = models.resnet152(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_cls)
    #model.fc = torch.nn.Sequential(torch.nn.Linear(model.fc.in_features, num_cls), torch.nn.Sigmoid())
    return model


def get_resnet50():
    num_cls = len(opt.attributes)
    model = models.resnet50(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_cls)
    return model


def get_resnet34():
    num_cls = len(opt.attributes)
    model = models.resnet34(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_cls)
    return model


def get_resnet18():
    num_cls = len(opt.attributes)
    model = models.resnet18(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, num_cls)
    return model


def get_vgg16():
    num_cls = len(opt.attributes)
    model = models.vgg16(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    # model.avgpool = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3),
    #                                nn.MaxPool2d(2),
    #                                nn.ReLU(),
    #                                nn.Flatten())
    # model.classifier = MobilityClassifier()
    model.classifier[-1] = torch.nn.Linear(4096, num_cls)
    return model


def get_vgg16_bn():
    num_cls = len(opt.attributes)
    model = models.vgg16_bn(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier[-1] = torch.nn.Linear(4096, num_cls)
    return model


def get_mobilnet():
    num_cls = len(opt.attributes)
    model = models.mobilenet_v3_large(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[-1] = torch.nn.Linear(1280, num_cls)
    return model

def get_vit_b_16():
    num_cls = len(opt.attributes)
    model = models.vit_b_16(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.heads = torch.nn.Linear(768, num_cls)
    return model

def get_vit_l_16():
    num_cls = len(opt.attributes)
    model = models.vit_l_16(weights=opt.weights)
    if opt.freeze:
        for param in model.parameters():
            param.requires_grad = False

    model.heads = torch.nn.Linear(1024, num_cls)
    return model

def get_model():
    if opt.model == 'densenet201':
        model = get_densenet201()
    elif opt.model == 'resnet152':
        model = get_resnet152()
    elif opt.model == 'resnet50':
        model = get_resnet50()
    elif opt.model == 'resnet34':
        model = get_resnet34()
    elif opt.model == 'resnet18':
        model = get_resnet18()
    elif opt.model == 'vgg16':
        model = get_vgg16()
    elif opt.model == 'vgg16_bn':
        model = get_vgg16_bn()
    elif opt.model == 'mobilnet':
        model = get_mobilnet()
    elif opt.model == 'vit_b_16':
        model = get_vit_b_16()
    elif opt.model == 'vit_l_16':
        model = get_vit_l_16()
    else:
        print(f'Model "{opt.model}" is not supported')
        sys.exit(1)

    return model
