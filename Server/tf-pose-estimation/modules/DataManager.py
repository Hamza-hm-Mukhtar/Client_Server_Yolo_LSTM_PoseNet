from threading import Thread
import cv2
import struct
import pickle
import tensorflow as tf
import numpy as np
import time


tf.config.set_visible_devices([], 'GPU')
config = tf.compat.v1.ConfigProto()
graph = tf.compat.v1.get_default_graph()
first_session = tf.compat.v1.Session(config=config)
with graph.as_default(), first_session.as_default():
    with graph.as_default():
        with tf.device('CPU:0'):
            model = tf.keras.models.load_model('/home/dilshod/Code/Server/tf-pose-estimation/lstm_model/model1.h5', compile=False)
print(model.summary())


def make_prediction(m):
    WINDOW_SIZE, alpha, theta = 5, 0.9, 3
    forecast_ewma, forecast_values, theta_values, arr_of_num = [0], [], [], [1,1,1,1,1]
    arr_of_num.append(m)
    if len(arr_of_num)>WINDOW_SIZE:
        arr_of_num = arr_of_num[1:]
    if len(arr_of_num)==WINDOW_SIZE:
        actual = arr_of_num[-1]
        with graph.as_default(), first_session.as_default():
            forecast = model.predict(np.array(arr_of_num[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1))[0][0]
            forecast_values.append(forecast)
            a = alpha * forecast + (1 - alpha) * forecast_ewma[-1]
            theta += 1 if a > 0.5 else -1
            theta = min(max(theta, 0), 2)
            theta_values.append(theta)
            forecast_ewma.append(a)
            return actual, forecast

class DataManagerThread(Thread):
    def __init__(self, queue,sock, index):  
        super().__init__()
        
        self.image_queue = queue
        self.server_socket = sock  
        self.index = index
        
        
    def run(self):
        data = b""
        payload_size = struct.calcsize("Q")
        while True:
            while len(data) < payload_size:
                packet = self.server_socket.recv(4*1024)
                data += packet
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack("Q", packed_msg_size)[0]
            
            while len(data) < msg_size:
                data += self.server_socket.recv(4*1024)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            data_dict = pickle.loads(frame_data)
            # extract frame and detection information from data dictionary
            img = data_dict['frame']
            people = data_dict['people'] 
            print(f'Detected number of people: {people}')
            self.put_data_to_queue(img)
            pred = make_prediction(people)
            print(f"Predictions: {pred}")

    def put_data_to_queue(self, image):
        self.image_queue.put(image)
