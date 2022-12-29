# MRI_tumour_detection
CNN neural network to classify MRI brain images to positive/negative for tumour 

## **Data**

I used data from two sources on Kaggle.
One with [250](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) and the second with [3000](https://www.kaggle.com/datasets/abhranta/brain-tumor-detection-mri)
images of brain MRI images with or without tumors.

## **Data preprocessing**

The images were cropped in order to reduce background noise on the edges of the brain by finding the edges of the brain.
First I performed Gaussian blur on the image to burr edges unrelated on the brain outer border (there are a lot of lines and contours in the brain)
Next, the image was thresholded in order to clear out the black background.
finally, the image was cropped according to the the edges of the largest contour (the brain).

## **Data split and augmentations**

The total 3000 cropped images were split to train (1920), validation (480) and test (600) subsets.
Data was generated with tensorflow ImageDataGenerator with augmentations of shearness, brightness, horizontal flip and vertical_flip.

## **Model**

A tensorflow sequential model with two convolutional layers with relu activation function followed by a pooling layer (X3) was generated.
Training was performed with binary cross entropy loss and Adam optimizer (with default learning rate of 0.001) for 18 epochs (the hyperparameters were chosen by trial and error)

Accuracy typically maxed at 90%.
