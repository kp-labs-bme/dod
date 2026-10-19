import os
import pickle
import shutil
import random
import tarfile
from PIL import Image
from types import SimpleNamespace
from torchvision import transforms
from data.download.download_cifar100 import download_cifar100
from utils.augmentation import setup_augmentation, heavy_augmentation


def unpickle(file: str) -> dict:
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def save_images(data, labels, label_names, folder_path):
    """
    Save CIFAR-100 data as images sorted by class into subfolders.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for idx, (img_data, label) in enumerate(zip(data, labels)):
        class_name = label_names[label]
        class_folder = os.path.join(folder_path, class_name)

        # Create class folder if it doesn't exist
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # Reshape the flat image data and save it as an image
        img = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)

        # Save the image
        img_path = os.path.join(class_folder, f'{idx}.png')
        img.save(img_path)


def get_image_paths_and_labels_from_folder(folder_path, label_mapping):
    """
    Collect image file paths from a folder organized by class subfolders and map them to labels.
    
    :param folder_path: Path to the folder containing class subfolders with images.
    :param label_mapping: Dictionary mapping class folder names to label indices.
    :return: List of image file paths and their corresponding labels.
    """
    image_paths = []
    labels = []
    for class_folder in os.listdir(folder_path):
        full_class_folder = os.path.join(folder_path, class_folder)
        if os.path.isdir(full_class_folder):
            label = label_mapping.get(class_folder)
            if label is None:
                print(f"Warning: No label found for class folder {class_folder}")
                continue
            for img_name in os.listdir(full_class_folder):
                image_paths.append(os.path.join(full_class_folder, img_name))
                labels.append(label)
    return image_paths, labels


def contaminate_dataset(train_image_paths, train_labels, label_names, n_duplicate, n_undoable, cifar100_folder, config):
    """
    Contaminates the dataset by duplicating and heavily augmenting (undoable) images.
    The originals of heavily augmented images are removed from the training set for the current run.
    When `config.data_loader.use_selection` is True, previously generated duplicate and undoable images are reused.
    """
    print(f"Initial training data size: {len(train_image_paths)}, Initial labels size: {len(train_labels)}")
    
    duplicate_dir = os.path.join(cifar100_folder, 'duplicate')
    undoable_dir = os.path.join(cifar100_folder, 'undoable')

    # Debugging print to show a few initial image-label pairings from the train set
    print("Initial training data and labels (first 5 samples):")
    for i in range(min(5, len(train_image_paths))):  # Print first 5 pairs as an example
        print(f"Image: {train_image_paths[i]}, Label: {train_labels[i]} ({label_names[train_labels[i]]})")

    if config.data_loader.use_selection:
        # Reuse previously generated duplicate and undoable data
        print("Reusing previously generated duplicate and undoable images")

        # Load duplicate images
        if os.path.exists(duplicate_dir):
            print(f"Loading images from duplicate directory: {duplicate_dir}")
            duplicate_count = 0
            for class_folder in os.listdir(duplicate_dir):
                class_folder_path = os.path.join(duplicate_dir, class_folder)
                for image_file in os.listdir(class_folder_path):
                    image_path = os.path.join(class_folder_path, image_file)
                    label = label_names.index(class_folder)  # Get label from class folder name
                    train_image_paths.append(image_path)
                    train_labels.append(label)
                    duplicate_count += 1
            print(f"Total duplicate images loaded: {duplicate_count}")
            # Print a few examples from the duplicate directory
            print("Sample duplicate data-label pairs:")
            for i in range(min(5, duplicate_count)):
                print(f"Image: {train_image_paths[-(i+1)]}, Label: {train_labels[-(i+1)]} ({label_names[train_labels[-(i+1)]]})")

        # Load undoable images and remove the originals (in memory)
        if os.path.exists(undoable_dir):
            original_indices_to_remove = []
            print(f"Loading images from undoable directory: {undoable_dir}")
            undoable_count = 0
            for class_folder in os.listdir(undoable_dir):
                class_folder_path = os.path.join(undoable_dir, class_folder)
                for image_file in os.listdir(class_folder_path):
                    undoable_image_path = os.path.join(class_folder_path, image_file)
                    original_filename = image_file.replace('undoable_', '')

                    # Find the original image path and mark it for removal
                    for idx, image_path in enumerate(train_image_paths):
                        if os.path.basename(image_path) == original_filename:
                            original_indices_to_remove.append(idx)
                            break

                    # Add undoable images to training data
                    label = label_names.index(class_folder)  # Get label from class folder name
                    train_image_paths.append(undoable_image_path)
                    train_labels.append(label)
                    undoable_count += 1
            print(f"Total undoable images loaded: {undoable_count}")
            # Print a few examples from the undoable directory
            print("Sample undoable data-label pairs:")
            for i in range(min(5, undoable_count)):
                print(f"Image: {train_image_paths[-(i+1)]}, Label: {train_labels[-(i+1)]} ({label_names[train_labels[-(i+1)]]})")

            # Remove the originals of the heavily augmented images (in memory, not from disk)
            print(f"Removing {len(original_indices_to_remove)} original images.")
            train_image_paths = [path for i, path in enumerate(train_image_paths) if i not in original_indices_to_remove]
            train_labels = [label for i, label in enumerate(train_labels) if i not in original_indices_to_remove]
    
    else:
        # Remove the folders if they already exist, then recreate them for new data
        if os.path.exists(duplicate_dir):
            shutil.rmtree(duplicate_dir)
        if os.path.exists(undoable_dir):
            shutil.rmtree(undoable_dir)

        os.makedirs(duplicate_dir, exist_ok=True)
        os.makedirs(undoable_dir, exist_ok=True)

        # Step 1: Duplicating images with random labels
        duplicate_indices = []
        if n_duplicate > 0:
            print(f"Duplicating {n_duplicate} images with random labels")
            duplicate_indices = random.sample(range(len(train_image_paths)), n_duplicate)

            for idx in duplicate_indices:
                duplicate_image_path = train_image_paths[idx]
                original_label = train_labels[idx]

                # Debugging: Check original and new labels for a few duplicates
                if idx < 5:
                    print(f"Duplicating image: {duplicate_image_path}, Original label: {original_label} ({label_names[original_label]})")

                # Choose a random label different from the original
                possible_labels = list(set(train_labels) - {original_label})
                random_label = random.choice(possible_labels)

                if idx < 5:
                    print(f"New label for duplicate: {random_label} ({label_names[random_label]})")

                class_name = label_names[random_label]
                class_folder = os.path.join(duplicate_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)

                img = Image.open(duplicate_image_path)
                duplicate_image_filename = f'duplicate_{os.path.basename(duplicate_image_path)}'
                img.save(os.path.join(class_folder, duplicate_image_filename))

                # Append the duplicate to the training data
                train_image_paths.append(os.path.join(class_folder, duplicate_image_filename))
                train_labels.append(random_label)

        # Step 2: Creating undoable (heavily augmented) images
        undoable_indices = []
        if n_undoable > 0:
            print(f"Applying extreme augmentations to {n_undoable} images")
            available_indices = [
                i for i in range(len(train_image_paths)) 
                if i not in duplicate_indices and 'duplicate' not in train_image_paths[i]
            ]  # Exclude duplicated images

            undoable_indices = random.sample(available_indices, n_undoable)

            # Track the original indices to remove later
            original_indices_to_remove = []

            for idx in undoable_indices:
                undoable_image_path = train_image_paths[idx]
                original_label = train_labels[idx]

                # Debugging: Check original labels for a few undoable images
                if idx < 5:
                    print(f"Augmenting image: {undoable_image_path}, Original label: {original_label} ({label_names[original_label]})")

                class_name = label_names[original_label]
                class_folder = os.path.join(undoable_dir, class_name)
                os.makedirs(class_folder, exist_ok=True)

                img = Image.open(undoable_image_path)
                augmented_img = heavy_augmentation(img, config)  # Apply heavy augmentation

                # Convert the augmented image (tensor) back to a PIL image before saving
                augmented_img_pil = transforms.ToPILImage()(augmented_img)

                undoable_image_filename = f'undoable_{os.path.basename(undoable_image_path)}'
                augmented_img_pil.save(os.path.join(class_folder, undoable_image_filename))

                # Append the undoable to the training data
                train_image_paths.append(os.path.join(class_folder, undoable_image_filename))
                train_labels.append(original_label)

                # Add the original index to the list for removal
                original_indices_to_remove.append(idx)

            # Remove the originals of the heavily augmented images from the training data (in memory, not from disk)
            print(f"Removing {len(original_indices_to_remove)} original images.")
            train_image_paths = [path for i, path in enumerate(train_image_paths) if i not in original_indices_to_remove]
            train_labels = [label for i, label in enumerate(train_labels) if i not in original_indices_to_remove]

    # Debugging: Check final training data and label pairs for a few samples
    print(f"Final sample data-label pairs after contamination (including duplicates and augmentations):")
    for i in range(min(5, len(train_image_paths))):  # Print a few samples for debugging
        print(f"Image: {train_image_paths[i]}, Label: {train_labels[i]} ({label_names[train_labels[i]]})")

    print(f"Final training data size: {len(train_image_paths)}, Final labels size: {len(train_labels)}")
    
    return train_image_paths, train_labels


def load_cifar100(config: SimpleNamespace, data_dir: str):
    folder = "cifar-100-python"
    extracted_path = os.path.join(data_dir, folder)

    # Create file system directories for train and validation images
    cifar100_folder = os.path.join(data_dir, "cifar100")
    train_image_folder = os.path.join(cifar100_folder, "train")
    val_image_folder = os.path.join(cifar100_folder, "val")

    # Check if the dataset files are already extracted
    if not os.path.exists(os.path.join(extracted_path, 'train')):
        print("CIFAR-100 loader: Required files not found.")
        tar_path = os.path.join(data_dir, folder + '.tar.gz')

        # Check if the archive is present
        if not os.path.exists(tar_path):
            # Download and extract the dataset
            print("CIFAR-100 archive not found. Initiating download...")
            download_cifar100(data_dir)
        else:
            print("Archive found. Extracting files...")
            # Extract the archive
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=data_dir)
            print("Extraction complete.")
    else:
        print("CIFAR-100 dataset is already available and extracted.")

    # CIFAR Training File
    print("Cifar100 loader: Load training data")
    train_file = os.path.join(extracted_path, 'train')
    train_batch = unpickle(train_file)
    train_data = train_batch[b'data']
    train_labels = train_batch[b'fine_labels']

    # CIFAR Test File
    print("Cifar100 loader: Load test data")
    test_file = os.path.join(extracted_path, 'test')
    test_batch = unpickle(test_file)
    val_data = test_batch[b'data']
    val_labels = test_batch[b'fine_labels']

    # Label Names
    print("Cifar100 loader: Load labels")
    meta_file = os.path.join(extracted_path, 'meta')
    meta = unpickle(meta_file)
    interpretable_labels = [label.decode('utf-8') for label in meta[b'fine_label_names']]

    # If the train and val folders do not exist, create and save images
    if not os.path.exists(train_image_folder) or not os.path.exists(val_image_folder):
        print("Image folder structure not found. Creating and saving images...")

        # Remove existing directories if they are partially created (optional, for robustness)
        if os.path.exists(train_image_folder):
            shutil.rmtree(train_image_folder)
        if os.path.exists(val_image_folder):
            shutil.rmtree(val_image_folder)

        # Save training images
        print(f"Saving training images to {train_image_folder}...")
        save_images(train_data, train_labels, interpretable_labels, train_image_folder)

        # Save validation images
        print(f"Saving validation images to {val_image_folder}...")
        save_images(val_data, val_labels, interpretable_labels, val_image_folder)

        print("Image saving complete.")
    else:
        print("Image folder structure already exists. Using existing images.")

    # Transform setup
    print("Setting up data transformations")
    train_transforms, val_transforms = setup_augmentation(config)

    # Create a mapping of class folder names to label indices
    label_mapping = {label: idx for idx, label in enumerate(interpretable_labels)}

    # Collect image paths for train and validation datasets
    train_image_paths, train_labels = get_image_paths_and_labels_from_folder(train_image_folder, label_mapping)
    val_image_paths, val_labels = get_image_paths_and_labels_from_folder(val_image_folder, label_mapping)

    # Duplicate and augment images if needed
    n_duplicate = config.data_loader.n_duplicate
    n_undoable = config.data_loader.n_undoable

    if n_duplicate > 0 or n_undoable > 0:
        train_image_paths, train_labels = contaminate_dataset(
            train_image_paths=train_image_paths, 
            train_labels=train_labels, 
            label_names=interpretable_labels, 
            n_duplicate=n_duplicate, 
            n_undoable=n_undoable, 
            cifar100_folder=cifar100_folder, 
            config=config
        )

    # Return the paths to the images instead of raw data
    cifar_data = dict()
    cifar_data["train_data"] = train_image_paths  # Image paths
    cifar_data["train_labels"] = train_labels
    cifar_data["val_data"] = val_image_paths  # Image paths
    cifar_data["val_labels"] = val_labels
    cifar_data["train_transforms"] = train_transforms
    cifar_data["val_transforms"] = val_transforms
    cifar_data["interpretable_labels"] = interpretable_labels
    cifar_data["dataset_folder"] = os.path.join(cifar100_folder, 'cifar100')

    return cifar_data
