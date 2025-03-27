from setuptools import setup, find_packages

setup(
    name="EuroSHAP",
    version="0.1.0",
    description="Deep learning models on Eurosat data with SHAP explainability",
    author="Patrick Drew",
    author_email="patrickmdrew@gmail.com",
    packages=find_packages(),              # Automatically find packages in the repo (e.g., euroshap, euroshap.utils, etc.)
    install_requires=[                     # List any dependencies your project has
        "numpy",
        "pandas",
        "torch",
        "shap"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # or your chosen license
        "Operating System :: OS Independent",
    ],
)