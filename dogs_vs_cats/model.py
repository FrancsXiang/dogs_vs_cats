import tensorflow as tf


def inference(images, batch_size, num_classes, training=True):
    '''
    :param images: image batch 4D Tensor(batch_size,height,width,channel) tf.float32
    :param batch_size: batch size
    :param num_classes: number of classes to predict
    :return:logits 2D Tensor(batch_size,num_classes) tf.float32
    '''

    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                  trainable=training)
        biases = tf.get_variable(name='biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                 trainable=training)
        conv = tf.nn.conv2d(images, weights, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool2d(conv1, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                  trainable=training)
        biases = tf.get_variable(name='biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                 trainable=training)
        conv = tf.nn.conv2d(norm1, weights, [1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(conv, name=scope.name)

    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool2d(conv2, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME', name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    with tf.variable_scope('fc_1') as scope:
        reshape = tf.reshape(norm2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable(name='weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                  trainable=training)
        biases = tf.get_variable(name='biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                 trainable=training)
        fc_1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        kp1 = tf.nn.dropout(fc_1, keep_prob=0.5)

    with tf.variable_scope('fc_2') as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32),
                                  trainable=training)
        biases = tf.get_variable(name='biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                 trainable=training
                                 )
        fc_2 = tf.nn.relu(tf.matmul(kp1, weights) + biases, name=scope.name)
        kp2 = tf.nn.dropout(fc_2, keep_prob=0.5)

    with tf.variable_scope('sofmax_linear') as scope:
        weights = tf.get_variable(name='weights',
                                  shape=[128, num_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32),
                                  trainable=training)

        biases = tf.get_variable(name='biases',
                                 shape=[num_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1, dtype=tf.float32),
                                 trainable=training)
        logits = tf.add(tf.matmul(kp2, weights), biases, name='logits')

    return logits


def losses(logits, labels):
    '''
    :param logits: the predictions 2D Tensor[batch_size,num_classes]
    :param labels: the ground truth 1D Tensor[batch_size]
    :return: loss Tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                       name='cross_entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)

    return loss


def training(loss, learning_rate):
    '''
    :param loss: the loss of network
    :param learning_rate: the learning_rate of network
    :return: ops to minimize the loss
    '''
    with tf.variable_scope('train_steps') as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_steps = tf.Variable(0, trainable=False, name='global_steps')
        train_op = optimizer.minimize(loss, global_step=global_steps)
    return train_op


def evaluation(logits, labels):
    '''
    :param logits: the predictions 2D Tensor[batch_size,num_classes]
    :param labels: the ground truth 1D Tensor[batch_size]
    :return: the accuracy of predictions
    '''
    with tf.variable_scope('accuracy') as scope:
        predictions = tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64))
        acc = tf.reduce_mean(tf.cast(predictions, tf.float32), name='result')
        tf.summary.scalar(acc.name, acc)
    return acc
