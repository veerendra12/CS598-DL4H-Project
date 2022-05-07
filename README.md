# CS598-DL4H-Project

## 📦 Computational Requirements
This study needs a GPU minimum of 16GB RAM, minimum configuration of T4, P100 or higher is needed. The code, primarily as ipynb notebooks, have been developed using [Google colab](https://colab.research.google.com/) GPU environment. And it uses [Google drive](https://drive.google.com/) as storage service. However, the code can be easily tweaked to run on a different environment and different storage. Use [notebooks/Configuration.ipynb](notebooks/Configuration.ipynb) to change storage location of data sets and results.


## 📦 Data Download
This study uses following two sets of data sets:

1. **NIH CXR Benchmark Data Set:** This data set comprises 112,120 frontal-view X-ray images with 14 disease labels and resolution of 1024x1024.
   The data set is publicly available in [NIC Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737), which is split into 12 zip files, with overall compressed size of 50+ GB.
   Run [notebooks/DownloadNIHData.ipynb](notebooks/DownloadNIHData.ipynb) to download the data set. The notebook downloads the 12 zip files and also takes copy to [Google drive](https://drive.google.com/).


2. **Montgomery County X-Ray Data Set:** This data set comprises 138 posterior-anterior X-ray images


## 🚆 Model Training

Citation to the original paper
Link to the original paper’s repo (if applicable)
Dependencies
Data download instruction
Preprocessing code + command (if applicable)
Training code + command (if applicable)
Evaluation code + command (if applicable)
Pretrained model (if applicable)
Table of results (no need to include additional experiments, but main reproducibility result should be included)
