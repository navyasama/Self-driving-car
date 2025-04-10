import os
import tensorflow.compat.v1 as tf
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

tf.disable_v2_behavior()


class DataLogger:
    def __init__(self, logs_path):
        self.logs_path = logs_path
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def log_summary(self, summary, step):
        self.summary_writer.add_summary(summary, step)

    def close(self):
        self.summary_writer.close()


class Trainer:
    def __init__(self, model, log_dir, logger, l2_norm_const=0.001, learning_rate=1e-4):
        self.log_dir = log_dir
        self.l2_norm_const = l2_norm_const
        self.logger = logger

        self.session = tf.InteractiveSession()
        self.loss, self.train_step = self._build_training_ops(model, learning_rate)
        self.saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V2)
        self.session.run(tf.global_variables_initializer())

        self.merged_summary_op = tf.summary.merge_all()

    def _build_training_ops(self, model, learning_rate):
        train_vars = tf.trainable_variables()
        loss = tf.reduce_mean(tf.square(tf.subtract(model.y_, model.y))) + \
               tf.add_n([tf.nn.l2_loss(v) for v in train_vars]) * self.l2_norm_const
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        tf.summary.scalar("loss", loss)
        return loss, train_step

    def train(self, epochs, batch_size):
        for epoch in range(epochs):
            self._train_one_epoch(epoch, batch_size)
            print(f"Epoch {epoch+1}/{epochs} completed.")

    def _train_one_epoch(self, epoch, batch_size):
        for i in range(int(driving_data.num_images / batch_size)):
            xs, ys = driving_data.LoadTrainBatch(batch_size)
            self.train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})

            if i % 10 == 0:
                self._log_progress(epoch, i, batch_size)

            if i % batch_size == 0:
                self._save_checkpoint()

    def _log_progress(self, epoch, step, batch_size):
        xs, ys = driving_data.LoadValBatch(batch_size)
        loss_value = self.loss.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value}")

        summary = self.merged_summary_op.eval(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 1.0})
        self.logger.log_summary(summary, epoch * driving_data.num_images / batch_size + step)

    def _save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        checkpoint_path = os.path.join(self.log_dir, "model.ckpt")
        self.saver.save(self.session, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    def close(self):
        self.session.close()
        self.logger.close()


if __name__ == "__main__":
    LOGDIR = 'model_training/train_steering_angle/save'
    LOGS_PATH = 'model_training/train_steering_angle/logs'
    EPOCHS = 30
    BATCH_SIZE = 100

    logger = DataLogger(LOGS_PATH)
    trainer = Trainer(model, LOGDIR, logger)

    try:
        trainer.train(EPOCHS, BATCH_SIZE)
    finally:
        trainer.close()

    print("Run the command line:\n" \
          "--> tensorboard --logdir=model_training/train_steering_angle/logs " \
          "\nThen open http://0.0.0.0:6006/ into your web browser")
