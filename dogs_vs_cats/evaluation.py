'''
if  and only if you have the labels already,do the test for performance of your network.haha.
'''
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
MAX_STEPS = 1000

eval_log_dir = './log/eval/'
eval_dir = './test/'
check_point_path = './log/model/'
eva, eva_labels = input_data.get_files(eval_dir)
eval_batch, eval_label_batch = input_data.get_batch(eva, eva_labels, IMG_H, IMG_W, BATCH_SIZE, CAPACITY)
eval_logits = model.inference(eval_batch, BATCH_SIZE, N_CLASSES, training=False)
eval_acc = model.evaluation(eval_logits, eval_label_batch)
eval_loss = model.losses(eval_logits, eval_label_batch)
summery_op = tf.summary.merge_all()


def run_testing():
    '''
    whole datasets testing
    :return: NONE
    '''
    TOTAL_ACC_SUM = 0
    TOTAL_LOSS_SUM = 0
    with tf.Session() as sess:
        eval_writer = tf.summary.FileWriter(eval_log_dir, sess.graph, session=sess)
        saver = tf.train.Saver()
        if os.path.exists(os.path.join(check_point_path, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(check_point_path))
        else:
            print('There is no checkpoint!')
            return

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for step in range(MAX_STEPS):
                if coord.should_stop(): break
                test_acc, test_loss = sess.run([eval_acc, eval_loss])
                if step % 100 == 0:
                    print('step %d The testing acc and loss respectively: %.2f %.2f' % (step, test_acc, test_loss))
                    summary_total = sess.run(summery_op)
                    eval_writer.add_summary(summary_total, global_step=step)
                TOTAL_ACC_SUM += test_acc
                TOTAL_LOSS_SUM += test_loss
        except tf.errors.OutOfRangeError:
            print('testing done!')
        finally:
            coord.request_stop()
    coord.join(threads)
    print('The average testing acc and loss respectively:%.2f' %
          ((TOTAL_ACC_SUM / MAX_STEPS), (TOTAL_LOSS_SUM / MAX_STEPS)))

run_testing()
