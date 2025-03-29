import requests
from torch.utils.data import Dataset
import torch


def report_filesize(url):
    response = requests.head(url)
    filesize = int(response.headers.get("Content-Length", 0))
    print("File size in MB:", filesize / (1024*1024))
    return


def shape_of_cutout(dataset: Dataset, index: int=0) -> None:
    """
    Print the shape of the first image in the dataset

    Args:
        dataset (pytorch Dataset): The dataset to analyze. Each element 
            is expected to be a tuple (image, label).
        index (int): the index of the image to be reported. defaults to 0

    Returns:
        None
    """
    img, _ = dataset[index]
    shape = list(img.shape)
    print(f'Shape of images [color, height, width]: {shape}')
    return None


def get_device() -> str:
    """
    Returns the type of accelerator device if available, otherwise returns 'cpu'.
    Also prints the device being used.
    
    Returns:
        str: The accelerator device type or 'cpu'.
    """
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    return device
