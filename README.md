# Alzheimer Research
Authors:
- Maritrini García [maritrini-gar](https://github.com/maritrini-gar)
- Mariano Villafuerte | [VillafuerteM](https://github.com/VillafuerteM)

This repository aims to document all the code done for the Alzheimer research. 

## Table of contents
- [Objective](#objective)
- [dlib installation](#dlib-installation)
- [DNN Module](#dnn-module)
- [Repository structure](#repository-structure)
- [Environment](#environment)

## Objective
This repository contains the necessary code to process the data gathered for this research

## dlib installation
This project uses Python 3.8 and dlib. For the installation:

- A new blank environment was created with Python 3.8 (face_recognition seemed to work properly on this version)
- Downloaded the dlib wheel from this [link](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)
- We place the wheel in the root of the environment.
- Installed the wheel with the command.
```python
python -m pip install dlib-19.22.99-cp38-cp38-win_amd64.whl
```
- Installed the face_recognition library with the command
```python
pip install face_recognition
```
- The rest of the libraries can be installed as usual with conda or pip

## DNN Module
As the base model of face_recognition did not suffice to solve the blurring problem, a DNN model was used. For this, we need to download two files:
- The model file: [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
- The weights file: [res10_300x300_ssd_iter_140000.caffemodel](https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel)
- Crate and place the files in the folder "models"

## Repository structure
The repository is structured as follows:
- **code**: 
    - Contains the code used for the research
- **data**: 
    - Contains the data used for the research
- **dlib**: 
    - Contains the dlib installation wheel
- **environment**
    - Contains the environment for documentation
- **models**: 
    - Contains the models used for the face detection

The folders are structured as follows:
```
D:.
├───code
│   └───previous_versions
├───data
│   ├───final
│   ├───processed
│   └───raw
│       ├───Fase Avanzada a
│       ├───Fase inicial
│       ├───Fase inicial B
│       └───Fase intermedia
├───dlib
├───environment
└───models
```

## Environment
The environment used for this project is documented in the file environment.yml. To create the environment, run the following command:
```python
conda env create -f environment.yml
```

Or you can use the requirements.txt file to install the necessary libraries with the following command:
```python
pip install -r requirements.txt
```

Note: the environment uses Python 3.8 as the base version. If a newer version is used, the dlib installation might not work properly. Specifically, the wheel file used for the installation is for Python 3.8.