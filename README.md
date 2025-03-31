# EuroSHAP

EuroSHAP is designed to investigate how effectively machine learning models highlight important pixels in the classification of [EuroSAT](https://github.com/phelber/EuroSAT) data. EuroSAT is an open source land use and land cover classification dataset that uses Sentinel-2 satellite images. The project aims to ensure that models not only achieve high classification accuracy but also provide meaningful insights into the decision-making process. Below is a summary of the project:

1) **Exploratory Data Analysis (EDA)**: Gain insights into data quality and structure through image statistics, outlier detection, file size verification, and visualization.
2) **Model Training with PyTorch**: Develop and train ML models on a training subset of the EuroSAT dataset.
3) **Model Evaluation and Selection**: Evaluate the quality of each model and select the best.
4) **Interpretability with SHAP** Apply `shap.DeepExplainer` to highlight pixels that contribute to the classification of each image, verifying the models are making predictions for the right reasons.

# Project Roadmap
Future work includes the following:

1) Write visualization and model evaluation routines
2) Write tests for functions already written
3) interpretability module with Deep SHAP
4) Add more models to compare with the baseline.
5) consider hyperparameter tuning as well as cross validation

# Summary of Work
Below is a summary of functions I've written in support of this project broken down by analysis type.
## Exploratory Data Analysis
### Computing Image Statistics
- Exploratory data analysis is important because, prior to exploration, one typically does not know much about the quality of the dataset. I perform some image statistics to look for images with channels that are either much brighter or much dimmer than the average because these may point to data issues that may cause modeling issues.
- The `image_stats` function iterates through a dataset using a PyTorch DataLoader to calculate the per-channel mean and standard deviation for each image. The function also allows this to be computed over a batch of several images if there are too many images to process in a reasonable time. These statistics are organized into a pandas DataFrame for ease of viewing. This function is demonstrated [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb).

### Detecting Outliers
- `outlier_report` uses `image_stats` to compute the per-image channel means as mentioned above and then calculate z-scores across the dataset for each channel. Images with channel means that have z-scores exceeding a specified threshold are flagged as outliers. These indices are returned in a dictionary. This function is demonstrated [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb).

### File Size Reporting:
- Prior to downloading the full dataset to my computer I wanted to check how large the dataset was. I wrote the `report_filesize` function to obtain the size of the zip file containing the dataset. This function is demonstrated [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb).

### Inspecting Image Dimensions
- The `shape_of_cutout` function reports on the shape of the given image. It returns the shape in the format [color, height, width]. This function is demonstrated [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb).

### Visualizing an Image
- The `plot_cutout` function plots a given image and its label from the dataset. This function converts the tensor format from [color, height, width] to [height, width, color] for visualization in matplotlib and then plots the image with the title showing its label. It can also save the plot if a save path is provided. This function is demonstrated [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb).

### Assess Balance of Label Classes
- During exploratory data analysis of classification data it is very important to understand how balanced or imbalanced your labels are. For instance, if you have two classes, e.g. forest and lake, and the vast majority of your dataset contains lakes, your ML models will likely have a hard time correctly identifying forests. This can be overcome via different methods, but a first step is to identify if your data is imbalanced. I investigate this [here](https://github.com/pdrew32/EuroSHAP/blob/dev/notebooks/exploratory_data_analysis.ipynb) and find the dataset has 10 classes all in roughly equal balance. Given this, I will not be correcting for imbalanced data.
