import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def get_files(file_dir):
    """
    :param file_dir: the location of datasets
    :return: list of imgs and labels
    """
    cats = []
    dogs = []
    label_cats = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    print('There are %d cats and %d dogs' % (len(cats), len(dogs)))
    img_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))
    # print(np.shape(img_list))
    temp = np.array([img_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    img_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return img_list, label_list


def get_batch(image, label, image_H, image_W, batch_size, capacity):
    """
    :param image: list type
    :param label: list type
    :param image_W: image width
    :param image_H: image height
    :param batch_size: batch size
    :param capacity: the maximum elements in queue
    :return:
        image_batch: 4D Tensor [batch_size,height,width,3] dtype(tf.float32)
        label_batch: 1D Tensor [batch_size] dtype(tf.float32)
    """
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image, image_H, image_W)
    image = tf.image.random_flip_left_right(image)
    # data augmentation
    # image = tf.image.random_brightness(image, max_delta=0.2)
    # image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
    # image = tf.image.random_saturation(image,lower=0.0, upper=2.0)
    image = tf.image.per_image_standardization(image)
    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)
    label_batch = tf.reshape(label_batch, [batch_size])
    return image_batch, label_batch


''' method call test
image_list, label_list = get_files(train_dir)
image_batch, label_batch = get_batch(image_list, label_list, img_height, img_width, batch_size, capacity)


with tf.Session() as sess:
    i = 0
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop() :
                img,label = sess.run([image_batch,label_batch])
                for j in np.arange(batch_size):
                    print('label:%d'%(label[j]))
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i += 1
    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()

    coord.join(threads)
'''
