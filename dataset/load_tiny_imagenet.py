import os
import shutil
import random
import numpy as np
from PIL import Image
from types import SimpleNamespace
from torchvision import transforms
from typing import Dict, Tuple, List
from utils.augmentation import setup_augmentation, heavy_augmentation
from data.download.download_tiny_imagenet import download_tiny_imagenet

def contaminate_dataset(train_data: np.ndarray, train_labels: np.ndarray, n_duplicate: int, n_undoable: int, dataset_root_dir: str, wnid_to_id: dict, config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Contaminates the dataset with n_duplicate randomly selected images with random labels, 
    and n_undoable images that are augmented to be unrecognizable. When config.data_loader.use_selection is True,
    previously generated duplicate and undoable data are reused.
    """
    # Paths to the new folders, placed at the same level as 'train'
    duplicate_dir = os.path.join(dataset_root_dir, 'duplicate')
    undoable_dir = os.path.join(dataset_root_dir, 'undoable')

    if config.data_loader.use_selection:
        # Load duplicate and undoable data from the previous run if the folders exist
        print(f"Reusing previously generated duplicate and undoable data")

        if os.path.exists(duplicate_dir):
            for class_folder in os.listdir(duplicate_dir):
                class_folder_path = os.path.join(duplicate_dir, class_folder)
                label_id = wnid_to_id[class_folder]

                for image_file in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_file)
                    train_data = np.append(train_data, image_path)
                    train_labels = np.append(train_labels, label_id)

        if os.path.exists(undoable_dir):
            for class_folder in os.listdir(undoable_dir):
                class_folder_path = os.path.join(undoable_dir, class_folder)
                label_id = wnid_to_id[class_folder]

                for image_file in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_file)
                    train_data = np.append(train_data, image_path)
                    train_labels = np.append(train_labels, label_id)

        return train_data, train_labels

    else:
        # Remove the folders if they already exist, then recreate them
        if os.path.exists(duplicate_dir):
            shutil.rmtree(duplicate_dir)
        if os.path.exists(undoable_dir):
            shutil.rmtree(undoable_dir)

        # Create directories for storing duplicate and undoable images if they don't exist
        os.makedirs(duplicate_dir, exist_ok=True)
        os.makedirs(undoable_dir, exist_ok=True)

        # Step 1: Select and duplicate images
        duplicate_indices = []
        if n_duplicate > 0:
            print(f"Duplicating {n_duplicate} images with random labels")
            duplicate_indices = np.random.choice(len(train_data), size=n_duplicate, replace=False)

            for idx in duplicate_indices:
                duplicate_image_path = train_data[idx]
                original_label = train_labels[idx]

                # Select a random label different from the original
                possible_labels = list(set(train_labels) - {original_label})
                random_label = np.random.choice(possible_labels)

                # Create the class directory for the random label in the 'duplicate' folder
                random_label_wnid = list(wnid_to_id.keys())[list(wnid_to_id.values()).index(random_label)]
                random_label_dir = os.path.join(duplicate_dir, random_label_wnid)
                os.makedirs(random_label_dir, exist_ok=True)

                # Load the image
                image = Image.open(duplicate_image_path)

                # Generate a proper filename for the duplicated image
                duplicate_image_filename = f"duplicate_{os.path.basename(duplicate_image_path)}"
                duplicate_image_save_path = os.path.join(random_label_dir, duplicate_image_filename)
                image.save(duplicate_image_save_path)

                # Add the duplicate image path and the new random label to the training set at the end
                train_data = np.append(train_data, duplicate_image_save_path)
                train_labels = np.append(train_labels, random_label)

        # Step 2: Select and augment (undoable) images, excluding the duplicated ones
        undoable_indices = []
        if n_undoable > 0:
            print(f"Applying extreme augmentations to {n_undoable} images")

            # Exclude the indices of duplicated images from the available pool
            available_indices = [i for i in range(len(train_data)) if i not in duplicate_indices]

            undoable_indices = np.random.choice(available_indices, size=n_undoable, replace=False)

            # Create a list to hold the indices of original images that will be removed
            original_indices_to_remove = []

            for idx in undoable_indices:
                original_image_path = train_data[idx]
                original_label = train_labels[idx]

                # Create the class directory for the original label in the 'undoable' folder
                original_label_wnid = list(wnid_to_id.keys())[list(wnid_to_id.values()).index(original_label)]
                original_label_dir = os.path.join(undoable_dir, original_label_wnid)
                os.makedirs(original_label_dir, exist_ok=True)

                # Load the image and apply extreme transformations
                image = Image.open(original_image_path)
                augmented_image = heavy_augmentation(image, config)  # Apply heavy augmentation with the config

                # Convert the augmented image (tensor) back to a PIL image before saving
                augmented_image_pil = transforms.ToPILImage()(augmented_image)

                # Generate a proper filename for the augmented image
                undoable_image_filename = f"undoable_{os.path.basename(original_image_path)}"
                undoable_image_save_path = os.path.join(original_label_dir, undoable_image_filename)
                augmented_image_pil.save(undoable_image_save_path)

                # Add the augmented image path to the training set at the end
                train_data = np.append(train_data, undoable_image_save_path)
                train_labels = np.append(train_labels, original_label)

                # Add the index of the original image to be removed later
                original_indices_to_remove.append(idx)

            # Remove the original images' access paths that were augmented
            if len(original_indices_to_remove) == config.data_loader.n_undoable:
                train_data = np.delete(train_data, original_indices_to_remove)
                train_labels = np.delete(train_labels, original_indices_to_remove)
            else:
                error_msg = "len(original_indices_to_remove) and config.data_loader.n_undoable should be the same"
                raise ValueError(error_msg)

    return train_data, train_labels

def load_meta_tiny_imagenet(root_dir: str) -> Tuple[Dict[str, int], Dict[int, str], list]:
    wnids_path = os.path.join(root_dir, 'tiny-imagenet-200', 'wnids.txt')
    words_path = os.path.join(root_dir, 'tiny-imagenet-200', 'words.txt')

    # Read the wnids.txt file to get the WordNet IDs
    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f]

    # Create a mapping from WordNet ID to a sequential integer ID
    wnid_to_id = {wnid: i for i, wnid in enumerate(wnids)}
    id_to_wnid = {i: wnid for i, wnid in enumerate(wnids)}

    # Read the words.txt file to get the mapping from WNID to human-readable labels
    words = {}
    with open(words_path, 'r') as f:
        for line in f:
            wnid, label = line.strip().split('\t')
            words[wnid] = label

    # Create a list of interpretable labels ordered by the integer ID
    interpretable_labels = [words[wnid] for wnid in wnids]

    return wnid_to_id, id_to_wnid, interpretable_labels


def get_tiny_imagenet_data(root_dir: str, freshly_downloaded: bool, config=None) -> Tuple:
    print("Getting Tiny ImageNet data")
    train_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'train')
    val_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'val')
    test_dir = os.path.join(root_dir, 'tiny-imagenet-200', 'test')

    # Ensure the directories exist
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Load interpretable labels
    wnid_to_id, id_to_wnid, interpretable_labels = load_meta_tiny_imagenet(root_dir)

    # Organize validation data into folders
    def organize_validation_data(val_dir, wnid_to_id):
        val_images_dir = os.path.join(val_dir, 'images')
        val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')

        if not os.path.exists(val_annotations_path):
            print(f"Validation annotations file not found at {val_annotations_path}")
            return

        with open(val_annotations_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_name, label_wnid = parts[0], parts[1]

                # Create class directory if it does not exist
                class_dir = os.path.join(val_dir, label_wnid)
                if not os.path.exists(class_dir):
                    os.makedirs(class_dir)

                # Move the image to the correct class directory
                src_path = os.path.join(val_images_dir, image_name)
                dst_path = os.path.join(class_dir, image_name)
                shutil.move(src_path, dst_path)

        # Remove the now empty 'images' directory
        shutil.rmtree(val_images_dir)

    if freshly_downloaded:
        organize_validation_data(val_dir, wnid_to_id)

    train_data = []
    train_labels = []

    print("Creating training data (paths) and training labels")
    for class_folder in os.listdir(train_dir):
        class_folder_path = os.path.join(train_dir, class_folder, 'images')

        if '.DS_Store' == class_folder:
            continue

        label_id = wnid_to_id[class_folder]

        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            train_data.append(image_path)
            train_labels.append(label_id)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)

    # Contaminate the training dataset or reuse contamination depending on the flag
    n_duplicate = config.data_loader.n_duplicate
    n_undoable = config.data_loader.n_undoable
    if n_duplicate > 0 or n_undoable > 0:
        # Move the contamination directories one level up from the train directory
        dataset_root_dir = os.path.dirname(train_dir)
        train_data, train_labels = contaminate_dataset(train_data=train_data, 
                                                       train_labels=train_labels, 
                                                       dataset_root_dir=dataset_root_dir,
                                                       wnid_to_id=wnid_to_id, 
                                                       config=config, 
                                                       n_duplicate=n_duplicate, 
                                                       n_undoable=n_undoable
                                                       )

    print("Creating validation data (paths) and validation labels")
    val_data = []
    val_labels = []

    # After organizing, update the validation paths to reflect the new structure
    for class_folder in os.listdir(val_dir):
        class_folder_path = os.path.join(val_dir, class_folder)

        # Skip files that aren't directories (e.g., annotation files)
        if not os.path.isdir(class_folder_path):
            continue

        label_id = wnid_to_id[class_folder]

        for image_file in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_file)
            val_data.append(image_path)
            val_labels.append(label_id)

    val_data = np.array(val_data)
    val_labels = np.array(val_labels)

    if len(val_data) != len(val_labels):
        print("Warning: The number of validation images does not match the number of labels!")

    return train_data, train_labels, val_data, val_labels, interpretable_labels


def load_tiny_imagenet(config: SimpleNamespace, data_dir: str) -> dict:
    if not os.path.exists(os.path.join(data_dir, 'tiny-imagenet-200')):
        download_tiny_imagenet(data_dir)
        freshly_downloaded = True
    else:
        freshly_downloaded = False

    print("Loading Tiny ImageNet data")
    train_data, train_labels, val_data, val_labels, interpretable_labels = get_tiny_imagenet_data(
        data_dir, freshly_downloaded, config)

    train_transforms, val_transforms = setup_augmentation(config)

    tiny_imagenet_data = dict()
    tiny_imagenet_data["train_data"] = train_data
    tiny_imagenet_data["train_labels"] = train_labels
    tiny_imagenet_data["val_data"] = val_data
    tiny_imagenet_data["val_labels"] = val_labels
    tiny_imagenet_data["train_transforms"] = train_transforms
    tiny_imagenet_data["val_transforms"] = val_transforms
    tiny_imagenet_data["interpretable_labels"] = interpretable_labels
    tiny_imagenet_data["dataset_folder"] = data_dir

    return tiny_imagenet_data
