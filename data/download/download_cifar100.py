import os
import tarfile
import requests
from io import BytesIO


def download_cifar100(path) -> None:
    print("Downloading CIFAR-100 Dataset")
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    response = requests.get(url, stream=True)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        extract_path = os.path.join(path)
        print(f"Extracting data to {extract_path}")

        # Opening the tar.gz file and extracting it
        tar = tarfile.open(mode="r:gz", fileobj=BytesIO(response.content))
        tar.extractall(path=extract_path)
        tar.close()

        print("Download and extraction complete!")
    else:
        print("Failed to download the dataset.")
        raise Exception("Failed to download CIFAR-100 dataset.")

