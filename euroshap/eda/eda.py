from torch.utils.data import DataLoader
import pandas as pd
from torch.utils.data import Dataset
import numpy as np

def image_stats(dataset: Dataset, batch_size: int=1) -> pd.DataFrame:
    """
    Compute the per-channel mean and standard deviation of images in the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A pytorch dataset object containing image data. Each image is expected to be 
        a tensor with shape (C, H, W) if batch_size is 1 or (B, C, H, W) if batch_size > 1.
    batch_size : int, optional
        The number of samples per batch. if batch_size > 1, the function computes
        the statistics for the batch across the 0th dimension. if batch_size == 1,
        it computes statistics for the single image. Must be at least 1. Defaults to 1.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two sets of columns:
            - columns named 'mean_ch{i}' for each channel's mean
            - columns named 'std_ch{i}' for each channel's standard deviation.
        Each row corresponds to a batch in the dataset.

    Raises
    ------
    ValueError
        if batch_size is < 1
    """
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    
    mean, std = [], []
    for img, _ in loader:
        # compute mean and std over H, W: inds 2 and 3
        if batch_size > 1:
            m = img.mean(dim=(0, 2, 3)).numpy()
            s = img.std(dim=(0, 2, 3)).numpy()
        elif batch_size == 1:
            m = img.mean(dim=(2, 3)).numpy()[0]
            s = img.std(dim=(2, 3)).numpy()[0]
        else: 
            raise ValueError('batch_size must be at least 1')
        mean.append(m)
        std.append(s)
    
    means = pd.DataFrame(data=mean, columns=[f"mean_ch{i}" for i in range(mean[0].shape[0])])
    stds = pd.DataFrame(data=std, columns=[f"std_ch{i}" for i in range(std[0].shape[0])])

    df = pd.merge(means, stds, left_index=True, right_index=True)
    return df


def outlier_report(dataset: Dataset, z_thresh: float = 3.0, batch_size: int=1) -> dict:
    """
    Generate an outlier report for per-channel image means in a dataset.
    
    This function calls `image_stats` with a batch size of 1 to compute per-image 
    statistics (mean and standard deviation for each channel). It then computes 
    the z-score for each image's mean in each channel relative to the overall distribution
    of that channel's means. Images with an absolute z-score above `z_thresh` are flagged 
    as outliers.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A PyTorch dataset where each sample is a tuple (image, label) and each image is 
        expected to be a tensor of shape (C, H, W).
    z_thresh : float, optional
        The z-score threshold beyond which an image is considered an outlier.
        Default is 3.0.
    batch_size : int, optional
        The number of images per calculation. defaults to 1.
    
    Returns
    -------
    dict
        A dictionary with channel indices as keys and numpy arrays of image indices (where 
        that channel's mean is considered an outlier) as values.
    """
    df = image_stats(dataset, batch_size=batch_size)
    
    # Determine number of channels from the DataFrame column names.
    n_channels = len([col for col in df.columns if col.startswith("mean_ch")])
    outlier_dict = {}
    
    # For each channel, compute z-scores and determine outliers.
    for ch in range(n_channels):
        channel_means = df[f"mean_ch{ch}"].values  # shape: (n_images,)
        global_mean = np.mean(channel_means)
        global_std = np.std(channel_means)
        z_scores = (channel_means - global_mean) / global_std
        outlier_inds = np.where(np.abs(z_scores) > z_thresh)[0]
        outlier_dict[f"mean_ch{ch}"] = outlier_inds
    
    return outlier_dict