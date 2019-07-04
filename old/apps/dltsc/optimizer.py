import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tempfile import TemporaryFile
import sys

path_project = "/home/igorcosta/soccer/"
sys.path.insert(1, path_project)

from core.config import PATH_JOBS_RESULTS
import pandas as pd

class Optimizer:

    def __init__(self, config, dataset, model):

        self.config = config
        self.dataset = dataset
        self.model = model

        # create a saver
        self.saver = tf.train.Saver(max_to_keep=1000)

        # create the loss function and the update rule
        self.loss, self.update_rule = None, None
        # number of test instances, <= batch size
        self.test_batch_size = tf.placeholder(shape=(), dtype=tf.int32)

        self.create_update_rule()

    # create the update rule
    def create_update_rule(self):

        def rps_value(y_true, y_pred):

            y_pred = tf.math.softmax(y_pred)

            sub1 = tf.math.subtract(tf.gather(y_pred, 0, axis=1), tf.gather(y_true, 0, axis=1))
            sub2 = tf.math.subtract(tf.gather(y_pred, 1, axis=1), tf.gather(y_true, 1, axis=1))
            op1 = tf.square(tf.math.add(sub1, sub2))
            op2 = tf.square(sub1)
            total = tf.scalar_mul(0.5,  tf.math.add(op1, op2))

            return tf.reduce_sum(total)

        def rps_loss(labels, predictions, weights=1.0, scope=None, loss_collection=ops.GraphKeys.LOSSES,
                     reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS):

            if labels is None:
                raise ValueError("labels must not be None.")
            if predictions is None:
                raise ValueError("predictions must not be None.")

            with ops.name_scope(scope, "rps",
                                (predictions, labels, weights)) as scope:
                predictions = math_ops.to_float(predictions)
                labels = math_ops.to_float(labels)
                predictions.get_shape().assert_is_compatible_with(labels.get_shape())
                losses = rps_value(predictions, labels)

            return losses

        with tf.name_scope("Optim"):
            # create the loss function
            if self.config['optim:loss_function'] == 'softmax_cross_entropy':
                self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.model.Y_batch,
                                                            logits=self.model.Y_hat)
            elif self.config['optim:loss_function'] == 'sigmoid_cross_entropy':
                self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.model.Y_batch,
                                                            logits=self.model.Y_hat)
            elif self.config['optim:loss_function'] == 'rps':
                self.loss = rps_loss(labels=self.model.Y_batch, predictions=self.model.Y_hat)

            # count the correct classification predictions in a batch
            self.predicted_labels = tf.argmax(self.model.Y_hat[:self.test_batch_size], 1, name="PredictedLabels")
            true_labels = tf.argmax(self.model.Y_batch[:self.test_batch_size], 1, name="TrueLabels")
            self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(self.predicted_labels, true_labels),
                                                             dtype=tf.float32),
                                                     name="CorrectPredictions")

            self.rps_train = rps_loss(labels=self.model.Y_batch, predictions=self.model.Y_hat)
            self.rps_test = rps_value(self.model.Y_batch[:self.test_batch_size], self.model.Y_hat[:self.test_batch_size])

            # get all the trainable variables
            trainable_vars = tf.trainable_variables()
            # apply the gradients using clipping to avoid their explosion
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                clipped_grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars),
                                                          self.config['optim:max_grad_norm'])

                self.update_rule = tf.train.AdamOptimizer(self.config['optim:eta']). \
                    apply_gradients(zip(clipped_grads, trainable_vars))


    # run the optimization routine
    def optimize(self):

        results = pd.DataFrame(columns=['epoch', 'loss', 'rps_train', 'accuracy', 'rps'])

        with tf.Session() as sess:

            # initialize all variables
            sess.run(tf.global_variables_initializer())
            loss = 0, 0
            freq=self.config['optim:progress_freqs']

            # iterate for a number of epochs
            for epoch_idx in range(self.config['optim:num_epochs']+1):

                # draw a random batch of instances
                X_batch, Y_batch = self.dataset.draw_batch(self.config['optim:batch_size'])

                # update the model to minimize the batch loss
                batch_loss = self.update_model(sess, X_batch, Y_batch)
                loss += batch_loss

                # print and loss and save the checkpoint of the model
                if epoch_idx % freq == 0:

                    if epoch_idx > 0:
                        loss /= freq

                    train_rps = self.train_classification_rps(sess)
                    test_acc, test_rps = self.test_classification_accuracy(sess)

                    print('DS', epoch_idx, self.dataset.dataset_name, loss, train_rps, test_acc, test_rps)
                    results.loc[len(results)] = [epoch_idx, loss, train_rps, test_acc, test_rps]

                    if (epoch_idx >= 5000) and (epoch_idx % 5000 == 0):
                        filename = PATH_JOBS_RESULTS + 'cnn_stats/' + self.config['title'] + '.csv'
                        print(filename)
                        results.to_csv(filename)
                        # self.saver.save(sess, "./saved_models/" + self.model.name + '_' + self.config['encoder_type']
                        #               + '_' + self.config['aggregation_type'] + "_" + self.dataset.dataset_name
                        #              + ".ckpt", global_step=epoch_idx//freq)

                    loss = 0

    # update the model for the pairs of similar and dissimilar series
    def update_model(self, sess, X_batch, Y_batch):

        tf.summary.FileWriter('/home/igorcosta/tensorlogs', sess.graph)

        # compute similarity and loss
        batch_loss = sess.run(self.loss, feed_dict={self.model.X_batch: X_batch,
                                                      self.model.Y_batch: Y_batch,
                                                      self.model.is_training: False})

        # update the deep similarity network
        sess.run(self.update_rule, feed_dict={self.model.X_batch: X_batch,
                                              self.model.Y_batch: Y_batch,
                                              self.model.is_training: True})

        return batch_loss

    # test results
    def test_classification_accuracy(self, sess):

        total_correct, total_instances, rps_correct = 0, 0, 0
        batch_size = self.config['optim:batch_size']

        # iterate through batches
        for batch_idx in range(0, self.dataset.num_test_instances, batch_size):

            X_batch = np.zeros(shape=(self.config['optim:batch_size'],
                                  self.config['dataset:length'],
                                  self.config['dataset:num_channels']))

            Y_batch = np.zeros(shape=(self.config['optim:batch_size'],
                                      self.config['dataset:num_classes']))

            end_idx, real_num_batch_instances = 0, 0
            if batch_idx + batch_size <= self.dataset.num_test_instances:
                end_idx = batch_idx + batch_size
                real_num_batch_instances = batch_size
            else:
                end_idx = self.dataset.num_test_instances
                real_num_batch_instances = self.dataset.num_test_instances - batch_idx

            #print(batch_idx, end_idx, real_num_batch_instances)

            X_batch[:real_num_batch_instances] = self.dataset.X_test[batch_idx:end_idx]
            Y_batch[:real_num_batch_instances] = self.dataset.Y_test[batch_idx:end_idx]

            correct_predictions, rps_predictions = sess.run([self.correct_predictions, self.rps_test], feed_dict={
                                            self.model.X_batch: X_batch,
                                            self.model.Y_batch: Y_batch,
                                            self.test_batch_size: real_num_batch_instances,
                                            self.model.is_training: False})

            #print(correct_predictions)

            total_correct += correct_predictions
            total_instances += real_num_batch_instances
            rps_correct += rps_predictions

        return total_correct/self.dataset.num_test_instances, rps_correct/self.dataset.num_test_instances


    def train_classification_rps(self, sess):

        total_rps, total_instances = 0, 0
        batch_size = self.config['optim:batch_size']
        count = 0

        # iterate through batches
        for batch_idx in range(0, self.dataset.num_train_instances, batch_size):

            X_batch = np.zeros(shape=(self.config['optim:batch_size'],
                                      self.config['dataset:length'],
                                      self.config['dataset:num_channels']))

            Y_batch = np.zeros(shape=(self.config['optim:batch_size'],
                                      self.config['dataset:num_classes']))

            end_idx, real_num_batch_instances = 0, 0

            if batch_idx + batch_size <= self.dataset.num_train_instances:
                end_idx = batch_idx + batch_size
                real_num_batch_instances = batch_size
            else:
                end_idx = self.dataset.num_train_instances
                real_num_batch_instances = self.dataset.num_train_instances - batch_idx

            #print(batch_idx, end_idx, real_num_batch_instances)

            X_batch[:real_num_batch_instances] = self.dataset.X_train[batch_idx:end_idx]
            Y_batch[:real_num_batch_instances] = self.dataset.Y_train[batch_idx:end_idx]

            rps_predictions = sess.run(self.rps_test, feed_dict={
                                            self.model.X_batch: X_batch,
                                            self.model.Y_batch: Y_batch,
                                            self.test_batch_size: real_num_batch_instances,
                                            self.model.is_training: False})


            total_rps = total_rps + rps_predictions
            total_instances += real_num_batch_instances
            count = count+1

        return total_rps/self.dataset.num_train_instances
