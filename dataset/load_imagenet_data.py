import os
import scipy
import random
import tarfile
from types import SimpleNamespace
from torchvision import transforms
from typing import Dict, Tuple, List


def load_meta(meta_path: str) -> Tuple[Dict[str, int], Dict[int, str], list]:
    # Load the meta.mat file
    meta = scipy.io.loadmat(meta_path)

    # The actual structure might vary based on the specific version of the dataset,
    # so one might need to adjust the following lines. Typically, the structure is named 'synsets'
    # and has an 'ILSVRC2012_ID' field and a 'WNID' (WordNet ID) field.

    synsets = meta['synsets']
    ids = [int(s['ILSVRC2012_ID'][0][0][0]) for s in synsets]  # Extract ILSVRC2012_ID
    wnids = [s["WNID"][0][0] for s in synsets]  # Extract WordNet ID (WNID)
    words = [s["words"][0][0] for s in synsets]

    # Create mutual mapping between WordNet ID to ILSVRC2012_ID
    wnid_to_id = dict(zip(wnids, ids))
    id_to_wnid = dict(zip(ids, wnids))

    return wnid_to_id, id_to_wnid, words


def get_imagenet_data(root_dir: str) -> Tuple[List[str], List[int], List[str], List[int], List[str]]:
    print("Getting imagenet data")
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    # Check if the devkit archive needs to be unpacked
    devkit_tar_file_path = os.path.join(root_dir, "ILSVRC2012_devkit_t12.tar.gz")
    if os.path.exists(devkit_tar_file_path):
        print("Unpack devkit archive")
        with tarfile.open(devkit_tar_file_path, 'r:gz') as archive:
            archive.extractall(root_dir)
        print("Devkit archive unpacked")

    # Check if the main training archive needs to be unpacked
    train_tar_file_path = os.path.join(root_dir, "ILSVRC2012_img_train.tar")
    if os.path.exists(train_tar_file_path):
        print("Unpack main training archive")
        with tarfile.open(train_tar_file_path, 'r') as archive:
            archive.extractall(train_dir)
        print("Finished unpacking main training archive")

        print("Unpacking all class-specific archives")
        for class_tar in os.listdir(train_dir):
            class_tar_path = os.path.join(train_dir, class_tar)
            class_folder = os.path.join(train_dir, class_tar.split('.')[0])  # Removing .tar extension

            if tarfile.is_tarfile(class_tar_path) and not os.path.exists(class_folder):
                os.makedirs(class_folder)
                with tarfile.open(class_tar_path, 'r') as archive:
                    archive.extractall(class_folder)
                os.remove(class_tar_path)
        print("Finished unpacking all class-specific archives")

    # Check if the validation archive needs to be unpacked
    val_tar_file_path = os.path.join(root_dir, "ILSVRC2012_img_val.tar")
    if os.path.exists(val_tar_file_path):
        print("Extracting validation files")
        with tarfile.open(val_tar_file_path, 'r') as archive:
            archive.extractall(val_dir)
        print("Finished unpacking validation archive")
    else:
        print(f"Validation tar file not found at {val_tar_file_path}")

    print("Creating wordnet/sysnet to int mapping")
    meta_path = root_dir + "/ILSVRC2012_devkit_t12/data/meta.mat"
    wnid_to_id, id_to_wnid, interpretable_labels = load_meta(meta_path)
    # Create a mapping for training folders to their respective labels using wnid_to_id
    folder_to_label_map = {folder: wnid_to_id[folder] for folder in os.listdir(train_dir) if folder in wnid_to_id}

    train_data = []
    train_labels = []

    print("Creating training data (paths) and training labels")
    for folder in os.listdir(train_dir):
        folder_path = os.path.join(train_dir, folder)
        if os.path.isdir(folder_path) and folder in folder_to_label_map:  # Ensure it's a valid class folder
            for image_file in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_file)
                train_data.append(image_path)
                train_labels.append(folder_to_label_map[folder])

    print("Creating validation data (paths) and validation labels")
    val_data = []
    val_labels = []

    # Load the paths of the validation images and sort them
    val_data = sorted([os.path.join(val_dir, image_file) for image_file in os.listdir(val_dir) if
                       image_file.endswith(('.JPEG', '.jpeg', '.jpg', '.JPG'))])

    # Load the validation labels
    ground_truth_file = os.path.join(root_dir, 'ILSVRC2012_devkit_t12', 'data',
                                     'ILSVRC2012_validation_ground_truth.txt')
    if not os.path.exists(ground_truth_file):
        print(f"Validation ground truth file not found at {ground_truth_file}")
    else:
        with open(ground_truth_file, 'r') as f:
            val_labels = [int(line.strip()) for line in f.readlines()]

    # Ensure the number of images and labels match
    if len(val_data) != len(val_labels):
        print("Warning: The number of validation images does not match the number of labels!")

    return train_data, train_labels, val_data, val_labels, interpretable_labels


def load_imagenet_data(config: SimpleNamespace, root_dir: str) -> dict:
    print("Loading ImageNet data")
    train_data, train_labels, val_data, val_labels, interpretable_labels = get_imagenet_data(root_dir)

    resize_param = 224

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transforms for training
    train_transforms = []
    train_transforms_temp = []
    val_transforms = []

    if bool(config.getint("augmentation", "on")):
        train_transforms.append(transforms.Resize((resize_param, resize_param), antialias=True))
        val_transforms.append(transforms.Resize((resize_param, resize_param), antialias=True))

        if bool(config.getint("augmentation", "crop")):
            train_transforms_temp.append(transforms.RandomApply([transforms.RandomResizedCrop(size=(resize_param,
                                                                                                    resize_param),
                                                                                              scale=(0.6, 0.95))],
                                                                p=config.getfloat("augmentation", "crop_probs")))
        if bool(config.getint("augmentation", "horizontal_flip")):
            train_transforms_temp.append(transforms.RandomHorizontalFlip(p=config.getfloat("augmentation", "hf_probs")))

        if bool(config.getint("augmentation", "color_jitter")):
            train_transforms_temp.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.5,
                                                                                        contrast=0.4,
                                                                                        saturation=0.3,
                                                                                        hue=0.2)],
                                                                p=config.getfloat("augmentation", "jitter_probs")))
        if bool(config.getint("augmentation", "blur")):
            train_transforms_temp.append(transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.2, 2.0))],
                                                                p=config.getfloat("augmentation", "blur_probs")))

        if bool(config.getint("augmentation", "adjust_sharpness")):
            train_transforms_temp.append(transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=3.0, p=0.5)],
                p=config.getfloat("augmentation", "sharpness_probs")))

        train_transforms.append(transforms.RandomApply(transforms=train_transforms_temp,
                                                       p=config.getfloat("augmentation", "probs")))

        train_transforms.append(transforms.ToTensor())
        val_transforms.append(transforms.ToTensor())

        if bool(config.getint("augmentation", "normalize")):
            train_transforms.append(transforms.Normalize(mean=mean, std=std))
            val_transforms.append(transforms.Normalize(mean=mean, std=std))

    else:
        train_transforms.append(transforms.Resize((resize_param, resize_param), antialias=True))
        train_transforms.append(transforms.ToTensor())

        val_transforms.append(transforms.Resize((resize_param, resize_param), antialias=True))
        val_transforms.append(transforms.ToTensor())

    train_transforms = transforms.Compose(train_transforms)
    val_transforms = transforms.Compose(val_transforms)

    imagenet_data = dict()
    imagenet_data["train_data"] = train_data
    imagenet_data["train_labels"] = train_labels
    imagenet_data["val_data"] = val_data
    imagenet_data["val_labels"] = val_labels
    imagenet_data["train_transforms"] = train_transforms
    imagenet_data["val_transforms"] = val_transforms
    imagenet_data["interpretable_labels"] = interpretable_labels

    return imagenet_data


class MultipleRandomErasing:
    def __init__(self, num_patches=4, p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0):
        """
        Apply RandomErasing multiple times randomly.
        :param num_patches: Number of patches to erase.
        :param p: Probability of erasing a single patch.
        :param scale: Proportion of image to erase per patch.
        :param ratio: Aspect ratio of the erase region.
        :param value: Erasing value.
        """
        self.num_patches = num_patches
        self.eraser = transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value)

    def __call__(self, img):
        for _ in range(self.num_patches):
            if random.uniform(0, 1) < self.eraser.p:
                img = self.eraser(img)
        return img
