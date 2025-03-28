import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_cutout(dataset: Dataset, index: int) -> None:
    """
    Plots a single image from the dataset at the specified index.

    This function retrieves an image and its corresponding label from the dataset,
    converts the image from a PyTorch tensor ([Color, Height, Width]) to a NumPy array ([Height, Width, Color])
    suitable for visualization, and displays it using matplotlib. The dataset is expected
    to have a 'classes' attribute that maps label indices to class names.

    Parameters:
        dataset (Dataset): A PyTorch dataset object that returns a tuple (image, label)
                           and has an attribute 'classes' for label names.
        index (int): The index of the image to plot.

    Returns:
        None
    """
    img, label = dataset[index]
    img_np = img.permute(1, 2, 0).numpy() # Convert from [Color, Height, Width] to [Height, Width, Color]

    plt.imshow(img_np)
    plt.title(f"Label: {dataset.classes[label]}")
    plt.axis('off')
    plt.show();


def shape_of_cutout(dataset: Dataset) -> None:
    """
    Print the shape of the first image in the dataset

    Args:
        dataset (pytorch Dataset): The dataset to analyze. Each element 
            is expected to be a tuple (image, label).

    Returns:
        None
    """
    img, _ = dataset[0]
    shape = list(img.shape)
    print(f'Shape of images [color, height, width]: {shape}')
    return None