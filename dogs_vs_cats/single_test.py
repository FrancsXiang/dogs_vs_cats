import os
import model
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

eval_dir = './test/'
check_point_path = './log/model/'

IMG_W = 208
IMG_H = 208
CHANNELS = 3
BATCH_SIZE = 1
NUM_CLASSES = 2


def get_one_image_name():
    return os.path.join(eval_dir, str(np.random.randint(1, 12501)) + '.jpg')


def run_single_test():
    '''
    using one image to test
    :return: NONE
    '''
    file_name = get_one_image_name()
    image_contents = tf.read_file(file_name, 'r')
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, IMG_H, IMG_W)
    image = tf.image.per_image_standardization(image)
    image = tf.reshape(image, [1, IMG_H, IMG_W, CHANNELS])
    logits = model.inference(image, BATCH_SIZE, NUM_CLASSES, training=False)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(check_point_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
        else:
            print('There is no checkpoint!')
            return
        res = sess.run(logits)
        if (np.argmax(res, 1) == 0):
            print('Prediction Result: cats')
        else:
            print('Prediction Result: Dogs')
    img = Image.open(file_name, 'r')
    plt.imshow(img)
    plt.show()

run_single_test()
