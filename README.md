# Client_Server_Yolo_LSTM_PoseNet
Person surveillance system

We consider socket transmission. The code consists of two parts: client and server side.

In client side, we perform yolo object detection, and it sends detected frames and detection info (number of people detected) to server side.

Server side consists of several tasks and performs them simultaneously. It receives frames and performs pose estimation as well as logs GPU utilization (the aim of this project is to monitor GPU utilization). When I add LSTM model implementation, only LSTM is predicting and about 84 % of GPU memory is occupied, despite pose estimation is just frozen.

Pre trained LSTM model is to make predictions by taking number of people as inputs (actually, we are receiving two types of data: frames are for pose estimation and detection info for LSTM prediction). 

Every client and server side should be run in different python environments and environment.yml files are provided in each server and client folder. 

1. To run pose estimation, first run ../Server/tf-pose-estimation/mains.py file in a terminal and then run ../Client/tf-pose-estimation/client.py file in another terminal with their environment. 

2. To run lstm model seperately, just run ../Server/tf-pose-estimation/lstm_test.py in a server environment.

3. To run both (lstm and pose estimation) models simultaneously, first uncomment [10, 37] and [71, 72] lines of code in ../Server/tf-pose-estimation/modules/DataMnager.py

!!! Code implemented in Tensorfllow 2.5 and on ubuntu 20.04

!!! Please consider that I want to run LSTM model on CPU and poes estimation model on GPU simultaneously.
