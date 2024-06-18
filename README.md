# Alzheimer Research
Author: Mariano Villafuerte (VillafuerteM)

This repository aims to document all the code done for the Alzheimer research. 

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