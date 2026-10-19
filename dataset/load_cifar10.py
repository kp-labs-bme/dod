import os
import pickle
import tarfile
import numpy as np
from utils.augmentation import setup_augmentation
from data.download.download_cifar10 import download_cifar10


def unpickle(file) -> dict:
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def extract_cifar10(tar_path, data_dir) -> None:
    """Extract the CIFAR-10 dataset tar.gz file."""
    print(f"Extracting CIFAR-10 dataset to {data_dir}")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")


def load_cifar10(config, data_dir) -> dict:
    folder = "cifar-10-batches-py"
    extracted_path = os.path.join(data_dir, folder)

    # Check if the dataset files are already extracted
    if not os.path.exists(os.path.join(extracted_path, 'data_batch_1')):
        print("CIFAR-10 dataset not found or incomplete. Checking for tar.gz file.")
        tar_path = os.path.join(data_dir, folder + '.tar.gz')

        # Download if tar.gz file also doesn't exist
        if not os.path.exists(tar_path):
            download_cifar10(data_dir)
    else:
        print("CIFAR-10 dataset is already extracted.")

    # CIFAR Training Files
    print("Cifar10 loader: Load training data")
    train_data = []
    train_labels = []
    for i in range(1, 6):
        train_file = os.path.join(extracted_path, f'data_batch_{i}')
        train_batch = unpickle(train_file)
        train_data.append(train_batch[b'data'])
        train_labels += train_batch[b'labels']
    train_data = np.concatenate(train_data, axis=0)

    # CIFAR Test File
    print("Cifar10 loader: Load test data")
    test_file = os.path.join(extracted_path, 'test_batch')
    test_batch = unpickle(test_file)
    val_data = test_batch[b'data']
    val_labels = test_batch[b'labels']

    # Label Names
    print("Cifar10 loader: Load labels")
    meta_file = os.path.join(extracted_path, 'batches.meta')
    meta = unpickle(meta_file)
    interpretable_labels = [label.decode('utf-8') for label in meta[b'label_names']]

    train_transforms, val_transforms = setup_augmentation(config)

    cifar_data = dict()
    cifar_data["train_data"] = train_data
    cifar_data["train_labels"] = train_labels
    cifar_data["val_data"] = val_data
    cifar_data["val_labels"] = val_labels
    cifar_data["train_transforms"] = train_transforms
    cifar_data["val_transforms"] = val_transforms
    cifar_data["interpretable_labels"] = interpretable_labels

    return cifar_data
