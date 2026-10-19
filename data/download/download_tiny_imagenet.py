import os
import requests
from zipfile import ZipFile
from io import BytesIO


def download_tiny_imagenet(path) -> None:
    print("Downloading Tiny Imagenet")
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    response = requests.get(url, stream=True)
    print(f"Response status: {response}")

    if response.status_code == 200:
        extract_path = os.path.join(path)
        print(f"Extracting data to {extract_path}")
        with ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall(extract_path)
        print("Download and extraction complete!")
    else:
        print("Failed to download the dataset.")
