import random
import time
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')

model = tf.keras.models.load_model('/home/dilshod/Cog/tf-pose-estimation/modules/lstm_model/model1.h5', compile=False)


def smooth_random(start, stop, step):
    return round(random.uniform(start, stop))

WINDOW_SIZE, alpha, theta = 5, 0.9, 3
arr_of_num = [1,1,1,1,1]

def make_prediction(m):
    global arr_of_num
    arr_of_num.append(m)
    if len(arr_of_num)>WINDOW_SIZE:
        arr_of_num = arr_of_num[1:]
    if len(arr_of_num)==WINDOW_SIZE:
        actual = arr_of_num[-1]
        forecast = model.predict(np.array(arr_of_num[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1))[0][0]
        return print(f'Actual: {actual}, Forecast: {forecast}')

while True:
    # Generate a random number
    n = smooth_random(0, 10, 1)
    print("Generated number: {}".format(n))
    make_prediction(n)
    time.sleep(1)
