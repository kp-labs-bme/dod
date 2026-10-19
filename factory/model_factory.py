import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights, ResNet50_Weights
from torchvision.models import mobilenet_v3_large, mobilenet_v3_small, resnet50
from torchvision.models import efficientnet_b0, efficientnet_b1, EfficientNet_B0_Weights, EfficientNet_B1_Weights

from factory.base_factory import BaseFactory


class ModelFactory(BaseFactory):
    @classmethod
    def create(cls, **kwargs) -> nn.Module:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = kwargs["config"]
        model = config.model.model_type
        pretrained = config.model.pretrained
        dataset = config.data_loader.data_folder_name

        if dataset == "Cifar10":
            num_classes = 10
        elif dataset == "Cifar100":
            num_classes = 100
        elif dataset == "Imagenet":
            num_classes = 1000
        elif dataset == "TinyImagenet":
            num_classes = 200
        else:
            raise NotImplementedError("Dataset can be 'Cifar10', 'Cifar100', 'TinyImagenet', 'Imagenet'")

        if model == "mobilenet_v3_large":
            my_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT if pretrained else None)

            if dataset == "Cifar10" or dataset == "Cifar100" or dataset == "TinyImagenet":
                # Modify the first convolution layer to adapt to low resolution datasets
                my_model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                          bias=False)  # Adjusting the output layer
            my_model.classifier[3] = nn.Linear(my_model.classifier[3].in_features, num_classes)

        elif model == "mobilenet_v3_small":
            my_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)

            if dataset == "Cifar10" or dataset == "Cifar100" or dataset == "TinyImagenet":
                # Modify the first convolution layer to adapt to low resolution datasets
                my_model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1),
                                                          bias=False)

            # Adjusting the output layer
            my_model.classifier[3] = nn.Linear(my_model.classifier[3].in_features, num_classes)

        elif model == "resnet50":
            my_model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)

            if dataset == "Cifar10" or dataset == "Cifar100" or dataset == "TinyImagenet":
                # Modify the first convolution layer to adapt to low resolution datasets
                my_model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
                # Modify the initial max pooling layer to prevent downsizing too fast
                my_model.maxpool = nn.Identity()

            # Adjusting the output layer
            my_model.fc = nn.Linear(my_model.fc.in_features, num_classes)

        elif model == "efficientnet_b0":
            my_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            # Adjusting the output layer
            my_model.classifier[1] = nn.Linear(my_model.classifier[1].in_features, num_classes)

            if dataset == "Cifar10" or dataset == "Cifar100" or dataset == "TinyImagenet":
                # Modify the first convolution layer to adapt to low resolution datasets
                my_model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        elif model == "efficientnet_b1":
            my_model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT if pretrained else None)
            # Adjusting the output layer
            my_model.classifier[1] = nn.Linear(my_model.classifier[1].in_features, num_classes)

            if dataset == "Cifar10" or dataset == "Cifar100" or dataset == "TinyImagenet":
                # Modify the first convolution layer to adapt to low resolution datasets
                my_model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)

        else:
            raise NotImplementedError(
                "Valid options: mobilenet_v3_large, mobilenet_v3_small, resnet50, efficientnet_b0, efficientnet_b1")

        my_model.to(device)

        return my_model
