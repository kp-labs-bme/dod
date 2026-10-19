import os
import requests
from io import BytesIO
import tarfile


def download_cifar10(path) -> None:
    print("Downloading CIFAR-10 Dataset")
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    response = requests.get(url, stream=True)
    print(f"Response status: {response.status_code}")

    if response.status_code == 200:
        extract_path = os.path.join(path)
        print(f"Extracting data to {extract_path}")

        tar = tarfile.open(mode="r:gz", fileobj=BytesIO(response.content))
        tar.extractall(path=extract_path)
        tar.close()

        print("Download and extraction complete!")
    else:
        print("Failed to download the dataset.")
