# CS598-DL4H-Project

## 📗 Summary
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

Due to computational challenges, our reproducibility study once considered 50% (randomly sampled) NIH data set. That constitutes:
* Training images: 44,848
* Validation images: 11,212

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

## Preprocessing of Data
The NIH label data [Data_Entry_2017_v2020.csv](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/data/Data_Entry_2017_v2020.csv) comes in a named format as below:

<img src="https://github.com/veerendra12/CS598-DL4H-Project/blob/main/media/RawLabelData.png" width=1000>

The preprocessing process converts into one hot vector form [NIH_CXR_Data_OneHot_Full.csv](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/data/NIH_CXR_Data_OneHot_Full.csv) as follows:

<img src="https://github.com/veerendra12/CS598-DL4H-Project/blob/main/media/SampleOneHotVector.png" width=1000>

For performing the preprocessing of the label data, perform:
1. Open [NIHPreprocessor.ipynb](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/NIHPreprocessor.ipynb) 
2. Run the method
```
convert_NIH_labels_to_onehotvectors()
```
All the image data sets are transformed during model runtime. No preprocessing before training is required.


## 🚆 Model Training

### Training Feature Extraction Model-1 (FEM-1):
FEM-1 uses Dense-121 architecture based model fed with full resolution of NIH CXR images for classifying 14 labels. Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb) before initiating the training:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | False           | Training mode                                                    |
| MODEL_TYPE                | FEM             | Train FEM type model                                             |
| USE_LUNG_REGION_GENERATOR | False           | Uses full CXR image for training                                 |
| SAMPLE_RATIO              | 0.5             | Uses 50% of the data set for train and validation.               |
| RUN_PREFIX                | fem-1_nih_50pc_ | Prefix for the model check point files                           |
| BATCH_SIZE                | 16              |                                                                  |
| NUM_EPOCHS                | 10              |                                                                  |
| RESUME_TRAINING           | False           | True if resuming a partial training session                      |
| EPOCH_START               | 0               | >0 when resuming a partial training session, from where to start |


For training FEM-1, perform:
1. Open [SDFNPipeLine.ipynb](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/SDFNPipeLine.ipynb) 
2. Run all

Once training is completed, take the best ``fem-1_nih_50pc_*i*.pth`` performing model for SDFN training later point. Update the ``fem-1_nih_50pc_*i*.pth`` in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb) 

| Parameter                 | Value                                                   |
|---------------------------|---------------------------------------------------------|
| FEM1_BEST_MODEL           | BASE_DIR + "results/fem-1_nih_50pc_*i*.pth"             |

### Training Feature Extraction Model-2 (FEM-2):
FEM-1 uses Dense-121 architecture based model fed with Lung Region cropped NIH CXR images for classifying 14 labels. This model uses earlier trained best *Lung Segmentation Model* for Lung segmentation creation. Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb) before initiating the training:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | False           | Training mode                                                    |
| MODEL_TYPE                | FEM             | Train FEM type model                                             |
| USE_LUNG_REGION_GENERATOR | True            | Uses Lung Region cropped NIH CXR images for training             |
| SAMPLE_RATIO              | 0.5             | Uses 50% of the data set for train and validation.               |
| RUN_PREFIX                | fem-2_nih_50pc_ | Prefix for the model check point files                           |
| BATCH_SIZE                | 16              |                                                                  |
| NUM_EPOCHS                | 10              |                                                                  |
| RESUME_TRAINING           | False           | True if resuming a partial training session                      |
| EPOCH_START               | 0               | >0 when resuming a partial training session, from where to start |


For training FEM-2, perform:
1. Open [SDFNPipeLine.ipynb](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/SDFNPipeLine.ipynb) 
2. Run all

Once training is completed, take the best ``fem-2_nih_50pc_*i*.pth`` performing model for SDFN training later point. Update the ``fem-2_nih_50pc_*i*.pth`` in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb)

| Parameter                 | Value                                                   |
|---------------------------|---------------------------------------------------------|
| FEM2_BEST_MODEL           | BASE_DIR + "results/fem-2_nih_50pc_*i*.pth"             |


### Training Segementation based Deep Fusion Netowrk (SDFN):
SDFN uses earlier pretrained best FEM-1 (``fem-1_nih_50pc_*i*.pth``) and FEM-2 (``fem-2_nih_50pc_*i*.pth``) to construct a Fusion network. SDFN gets the last layer latent vector for full CXR image from FEM-1 and Lung region image from FEM-2, concatenate the vectors and passes to a FC CNN for classification of the 14 Thoracic diseases. Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb) before initiating the training:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | False           | Training mode                                                    |
| MODEL_TYPE                | SDFN            | Train SDFNtype model                                             |
| USE_LUNG_REGION_GENERATOR | False           | FEM-1 uses full img and FEM-2 uses lung region img automatically |
| SAMPLE_RATIO              | 0.5             | Uses 50% of the data set for train and validation.               |
| RUN_PREFIX                | sdfn_nih_50pc_  | Prefix for the model check point files                           |
| BATCH_SIZE                | 16              |                                                                  |
| NUM_EPOCHS                | 10              |                                                                  |
| RESUME_TRAINING           | False           | True if resuming a partial training session                      |
| EPOCH_START               | 0               | >0 when resuming a partial training session, from where to start |

For training SDFN, perform:
1. Open [SDFNPipeLine.ipynb](https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/SDFNPipeLine.ipynb) 
2. Run all


## ⏱️ Model Evaluation
### Evaluating Feature Extraction Model-1 (FEM-1):
Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb), in additon to 'FEM-1 Training' parameters before initiating the evaluation run:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | True            | Evaluationode                                                    |

### Evaluating Feature Extraction Model-1 (FEM-2):
Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb), in additon to 'FEM-2 Training' parameters before initiating the evaluation run:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | True            | Evaluationode                                                    |


### Evaluating Segementation based Deep Fusion Netowrk (SDFN):
Ensure or set the following configuration parameters in Configuration.ipynb(https://github.com/veerendra12/CS598-DL4H-Project/blob/main/notebooks/Configuration.ipynb), in additon to 'SDFN Training' parameters before initiating the evaluation run:

| Parameter                 | Value           | Description                                                      |
|---------------------------|-----------------|------------------------------------------------------------------|
| EVAL_MODE                 | True            | Evaluationode                                                    |



## 🥅 Results
