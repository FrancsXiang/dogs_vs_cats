import os
import numpy as np
import tensorflow as tf
import input_data
import model

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 32
CAPACITY = 2000
MAX_STEPS = 10000
LEARNING_RATE = 0.0001
'''
you can use more complex model,but the bottleneck could be the datasets itself.haha
'''
def run_training():
    '''
    whole datasets training
    :return: NONE
    '''
    train_dir = './train/'
    logs_train_dir = './log/summary/'
    check_point_path = './log/model/'

    train, train_labels = input_data.get_files(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train, train_labels, IMG_H, IMG_W, BATCH_SIZE, CAPACITY)
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summery_op = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('./log/summary/', graph=sess.graph, session=sess)
        saver = tf.train.Saver(max_to_keep=1)
        if os.path.exists(os.path.join(check_point_path,'checkpoint')):
            saver.restore(sess,tf.train.latest_checkpoint(check_point_path))
        else:
            sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            for step in range(MAX_STEPS):
                if coord.should_stop(): break
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

                if step % 50 == 0:
                    print('The training loss and acc respectively: %.2f %.2f' % (tra_loss, tra_acc))
                    summary_total = sess.run(summery_op)
                    train_writer.add_summary(summary_total, global_step=step)

                if step % 2000 == 0 or (step + 1) == MAX_STEPS:
                    saver.save(sess, check_point_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('training done!')
        finally:
            coord.request_stop()
    coord.join(threads)


run_training()
