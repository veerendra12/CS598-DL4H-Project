# CS598-DL4H-Project

## Summary
This is a reproducibility study of multi label classification problem in Healthcare Industry proposed in following research paper. 

### Citation
> **Title:** SDFN Segmentation-based Deep Fusion Network for Thoracic Disease Classification in Chest X-ray Images

> **Authors:** Han Liu, Lei Wang, Yandong Nan, Faguang Jin, Qi Wang, and Jiantao Pu

> **Published On:** 30 Oct 2018
> **Journal:** Comput Med Imaging Graph, 2019

The key focus of the paper is to improve classification performance with a novel architecture proposed as **Segmentation-based Deep Fusion Network**. The SDFN architecture is based on the crucial idea that most of Thoracic disease related features are centered within lung region. So the network builds first a Lung segmentation model, with which a dual pathway CNN using DenseNet-121s are trained. One model (referred as Feature Extraction Model - FEM) with actual full CX image (FEM-1) and another model (FEM-2) fed with cropped Lung region image. The latent vectors from both FEM-1 and FEM-2 are taken to train final SDFN model to multi-classify among 14 Thoracic Lung diseases. The hypothesis of the paper is that SDFN using Lung Segementation improves performance over non-Lung Segementation based model.



### Model Pipeline Architecture
Here is the pipeline flow of different model sequence and their inter-dependency along with data sets.

<img src="https://github.com/veerendra12/CS598-DL4H-Project/blob/main/media/TrainingPipeline.png" width=600>


## ⚙️ Computational Requirements
This study needs a GPU minimum of 16GB RAM, minimum configuration of T4, P100 or higher is needed. The code, primarily as ipynb notebooks, have been developed using [Google colab](https://colab.research.google.com/) GPU environment. And it uses [Google drive](https://drive.google.com/) as storage service. However, the code can be easily tweaked to run on a different environment and different storage. Use [notebooks/Configuration.ipynb](notebooks/Configuration.ipynb) to change storage location of data sets and results.

*Configuration of the machine used:*
> GPU: Tesla P100-PCIE

> Memory: 16280MiB

> Storage: 200 GB

## 📦 Data Download
This study uses following two sets of data sets:

1. **NIH CXR Benchmark Data Set:** This is the primary data set for the study to build and a train multi-label disease classification based on a Chest X Ray (CXR) image. This data set comprises 112,120 frontal-view X-ray images with 14 disease labels and resolution of 1024x1024.
   The data set is publicly available in [NIC Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC/folder/37178474737), which is split into 12 zip files, with overall compressed size of 50+ GB.
   Run [notebooks/DownloadNIHData.ipynb](notebooks/DownloadNIHData.ipynb) to download the data set. The notebook downloads the 12 zip files and also takes copy to [Google drive](https://drive.google.com/).


2. **Montgomery County X-Ray Data Set:** This is the secondary data set used to build and train Lung Region Segmentation model[^1]. This data set comprises 138 posterior-anterior X-ray images along with corresponding greyscale mask images. The image resolution is 4020x4892 and overall size of 5+ GB. Download the data from [here](https://www.kaggle.com/code/eduardomineo/u-net-lung-segmentation-montgomery-shenzhen/data).

[^1]: The original paper used JSRT Dataset for Lung segmentation model training. Because of the challenge in getting the full dataset, we have used cited Montgomery Country X-Ray Data set for Lung Segmentation model 

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
