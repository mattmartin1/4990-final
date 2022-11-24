# Welcome to Vision Based Obstacle Detection How To Guide

Included in this demo are 3 methods we used to detect obstacles using computer vision.

### Method #1 - Sample Matching

##### To use this method load a sample into image 1 and a source into image 2.

From the provided files inside "SampleMatchDemo"
- Loading "box_cropped" into image 1 and "hall_with_box" into image 2 will result in a **success**
- Loading "pumpkin_cropped" into image 1 and "hall_with_box" into image 2 will result in a **failure**

### Method #2 - Trained Model Matching

##### To use this method load a source into image 2.

From the provided files inside "TrainedModelsDemo"
- Loading "hall_with_obstacles" into image 2 will result in a **success**
- Loading "no_obstacles" into image 2 will result in a **failure**

### Method #3 - Crop Based Obstacle Detection

##### To use this method load 2 similar images into images 1 and 2 (like 2 frames from a video)
From the provided files inside "CropDemo"
- Loading "clear1" and "clear2" will result in a **success**
- Loading "obstacle1" and "obstacle2" will result in a **success**

## Toolbar Menu

##### Tools
1. Reset GUI - this will refresh the GUI so you can start a new detection
2. Exit - this will close the program

##### Help
1. How To Guide - this will open the guide you are reading in a browser
2. GitHub Project - this will open the Git repository in a browser
3. Gain Access - this will display who to contact to gain access to the git repository
