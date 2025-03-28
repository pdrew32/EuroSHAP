import matplotlib.pyplot as plt
from torch.utils.data import Dataset


def plot_cutout(dataset: Dataset, index: int, save_path: str=None) -> None:
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
    if save_path:
        plt.savefig(save_path)
    plt.show();