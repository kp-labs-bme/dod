import os
from typing import Tuple
from types import SimpleNamespace
from factory.base_factory import BaseFactory
from dataset.custom_dataset import CifarDataset, TinyImagenetDataset, ImagenetDataset
from dataset.load_cifar import load_cifar
from dataset.load_tiny_imagenet import load_tiny_imagenet
from dataset.load_imagenet_data import load_imagenet_data


class DatasetFactory(BaseFactory):

    @classmethod
    def create(cls, config: SimpleNamespace) -> Tuple:
        data_folder_name = config.data_loader.data_folder_name

        root = os.getcwd()
        file_name = os.path.join(root, "data", data_folder_name)
        if data_folder_name == "Cifar100":
            training_dataset = load_cifar(config=config,
                                          data_dir=file_name,
                                          num_classes=100)
        elif data_folder_name == "Cifar10":
            training_dataset = load_cifar(config=config,
                                          data_dir=file_name,
                                          num_classes=10)
        elif data_folder_name == "TinyImagenet":
            training_dataset = load_tiny_imagenet(config=config,
                                                  data_dir=file_name)
        elif data_folder_name == "ImageNet":
            training_dataset = load_imagenet_data(file_name)
        else:
            raise NotImplementedError("Invalid dataset ('Cifar100' or 'Cifar10' or 'TinyImagenet' or 'ImageNet'")

        train_data = training_dataset["train_data"]
        train_labels = training_dataset["train_labels"]
        val_data = training_dataset["val_data"]
        val_labels = training_dataset["val_labels"]
        train_transforms = training_dataset["train_transforms"]
        val_transforms = training_dataset["val_transforms"]
        interpretable_labels = training_dataset["interpretable_labels"]
        root_folder = training_dataset["dataset_folder"]

        if data_folder_name == "ImageNet":
            # For training
            train_dataset = ImagenetDataset(data=train_data,
                                            labels=train_labels,
                                            transform=train_transforms,
                                            interpretable_labels=interpretable_labels)

            # For validation
            val_dataset = ImagenetDataset(data=val_data,
                                          labels=val_labels,
                                          transform=val_transforms)
        elif data_folder_name == "Cifar100" or data_folder_name == "Cifar10":
            # For training
            train_dataset = CifarDataset(data=train_data,
                                         labels=train_labels,
                                         transform=train_transforms,
                                         interpretable_labels=interpretable_labels)

            # For validation
            val_dataset = CifarDataset(data=val_data,
                                       labels=val_labels,
                                       transform=val_transforms)
        elif data_folder_name == "TinyImagenet":
            train_dataset = TinyImagenetDataset(data=train_data,
                                                labels=train_labels,
                                                transform=train_transforms,
                                                interpretable_labels=interpretable_labels)

            # For validation
            val_dataset = TinyImagenetDataset(data=val_data,
                                              labels=val_labels,
                                              transform=val_transforms)

        else:
            raise NotImplementedError("Not 'CIFAR' or 'ImageNet' or 'TinyImagenet'")

        return train_dataset, val_dataset, root_folder
