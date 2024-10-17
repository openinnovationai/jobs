from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import os
import sys
import tempfile
import time
import logging

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras.datasets import mnist

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")
logger = logging.getLogger(__name__)

flags = tf.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data", "Directory for storing MNIST data")
flags.DEFINE_boolean(
    "download_only",
    False,
    "Only perform downloading of data; do not proceed to "
    "session preparation, model definition, or training",
)
flags.DEFINE_integer("task_index", None, "Worker task index, should be >= 0")
flags.DEFINE_integer(
    "num_gpus",
    1,
    "Total number of GPUs for each machine. "
    "If you don't use GPU, please set it to '0'",
)
flags.DEFINE_integer(
    "replicas_to_aggregate",
    None,
    "Number of replicas to aggregate before applying parameter "
    "updates (for sync_replicas mode only; default: num_workers)",
)
flags.DEFINE_integer(
    "hidden_units", 100, "Number of units in the hidden layer of the NN"
)
flags.DEFINE_integer(
    "train_steps", 20000, "Number of (global) training steps to perform"
)
flags.DEFINE_integer("batch_size", 100, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean(
    "sync_replicas",
    False,
    "Use the sync_replicas (synchronized replicas) mode, wherein the parameter "
    "updates from workers are aggregated before applied to avoid stale gradients",
)
flags.DEFINE_boolean(
    "existing_servers",
    False,
    "Whether servers already exist. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow server.",
)
flags.DEFINE_string("job_name", None, "Job name: worker or ps")

FLAGS = flags.FLAGS

IMAGE_PIXELS = 28


def main(unused_argv):
    # Parse TF_CONFIG environment variable to get cluster configuration
    tf_config = json.loads(os.environ.get("TF_CONFIG", "{}"))
    task_config = tf_config.get("task", {})
    task_type = task_config.get("type")
    task_index = task_config.get("index")

    FLAGS.job_name = task_type
    FLAGS.task_index = task_index

    logger.info(f"Starting task: {FLAGS.job_name}, index: {FLAGS.task_index}")

    if FLAGS.download_only:
        logger.info("Downloading MNIST data...")
        # Download MNIST data
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        np.savez(
            os.path.join(FLAGS.data_dir, "mnist_data.npz"),
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
        )
        logger.info("MNIST data downloaded and saved.")
        sys.exit(0)

    if FLAGS.job_name is None or FLAGS.job_name == "":
        logger.error("Must specify an explicit `job_name` in TF_CONFIG.")
        raise ValueError("Must specify an explicit `job_name` in TF_CONFIG.")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        logger.error("Must specify an explicit `task_index` in TF_CONFIG.")
        raise ValueError("Must specify an explicit `task_index` in TF_CONFIG.")

    logger.info(f"Job name = {FLAGS.job_name}")
    logger.info(f"Task index = {FLAGS.task_index}")

    cluster_config = tf_config.get("cluster", {})
    ps_hosts = cluster_config.get("ps", [])
    worker_hosts = cluster_config.get("worker", [])

    if not ps_hosts or not worker_hosts:
        logger.error("TF_CONFIG cluster specification is missing or incomplete.")
        raise ValueError("TF_CONFIG cluster specification is missing or incomplete.")

    logger.info(f"Cluster specification: {cluster_config}")

    # Construct the cluster and start the server
    cluster = tf.train.ClusterSpec(cluster_config)

    if not FLAGS.existing_servers:
        # Create an in-process server
        logger.info(
            f"Starting in-process server for job: {FLAGS.job_name}, index: {FLAGS.task_index}"
        )
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index
        )
        if FLAGS.job_name == "ps":
            logger.info("Parameter server started, waiting for connections...")
            server.join()
            return

    is_chief = FLAGS.task_index == 0 and FLAGS.job_name == "worker"
    logger.info(f"Is chief: {is_chief}")

    if FLAGS.num_gpus > 0:
        gpu = FLAGS.task_index % FLAGS.num_gpus
        worker_device = f"/job:worker/task:{FLAGS.task_index}/gpu:{gpu}"
    else:
        worker_device = f"/job:worker/task:{FLAGS.task_index}/cpu:0"

    logger.info(f"Worker device: {worker_device}")

    with tf.device(
        tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster)
    ):
        # Build the model
        logger.info("Building the model...")
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # Define model variables
        hid_w = tf.Variable(
            tf.truncated_normal(
                [IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                stddev=1.0 / IMAGE_PIXELS,
            ),
            name="hid_w",
        )
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")
        sm_w = tf.Variable(
            tf.truncated_normal(
                [FLAGS.hidden_units, 10], stddev=1.0 / math.sqrt(FLAGS.hidden_units)
            ),
            name="sm_w",
        )
        sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

        # Define input placeholders
        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS], name="x")
        y_ = tf.placeholder(tf.int64, [None], name="y_")

        # One-hot encode labels
        y_one_hot = tf.one_hot(y_, depth=10)

        # Build the neural network
        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)
        logits = tf.nn.xw_plus_b(hid, sm_w, sm_b)
        y = tf.nn.softmax(logits)

        # Define loss and optimizer
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=logits)
        )

        logger.info("Setting up the optimizer...")
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)

        if FLAGS.sync_replicas:
            logger.info("Using SyncReplicasOptimizer...")
            replicas_to_aggregate = FLAGS.replicas_to_aggregate or len(worker_hosts)
            logger.info(f"Replicas to aggregate: {replicas_to_aggregate}")
            opt = tf.train.SyncReplicasOptimizer(
                opt,
                replicas_to_aggregate=replicas_to_aggregate,
                total_num_replicas=len(worker_hosts),
                name="mnist_sync_replicas",
            )

        train_step = opt.minimize(cross_entropy, global_step=global_step)

        if FLAGS.sync_replicas:
            local_init_op = opt.local_step_init_op
            if is_chief:
                local_init_op = opt.chief_init_op

            ready_for_local_init_op = opt.ready_for_local_init_op
            chief_queue_runner = opt.get_chief_queue_runner()
            sync_init_op = opt.get_init_tokens_op()

        init_op = tf.global_variables_initializer()
        train_dir = tempfile.mkdtemp()
        logger.info(f"Training directory: {train_dir}")

        # Set up the Supervisor
        logger.info("Setting up the Supervisor...")
        if FLAGS.sync_replicas:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                local_init_op=local_init_op,
                ready_for_local_init_op=ready_for_local_init_op,
                recovery_wait_secs=1,
                global_step=global_step,
            )
        else:
            sv = tf.train.Supervisor(
                is_chief=is_chief,
                logdir=train_dir,
                init_op=init_op,
                recovery_wait_secs=1,
                global_step=global_step,
            )

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=["/job:ps", f"/job:worker/task:{FLAGS.task_index}"],
        )

        if is_chief:
            logger.info(f"Worker {FLAGS.task_index}: Initializing session...")
        else:
            logger.info(
                f"Worker {FLAGS.task_index}: Waiting for session to be initialized..."
            )

        if FLAGS.existing_servers:
            server_grpc_url = f"grpc://{worker_hosts[FLAGS.task_index]}"
            logger.info(f"Using existing server at: {server_grpc_url}")
            sess = sv.prepare_or_wait_for_session(server_grpc_url, config=sess_config)
        else:
            sess = sv.prepare_or_wait_for_session(server.target, config=sess_config)

        logger.info(f"Worker {FLAGS.task_index}: Session initialization complete.")

        if FLAGS.sync_replicas and is_chief:
            logger.info("Chief worker: Initializing sync variables...")
            sess.run(sync_init_op)
            sv.start_queue_runners(sess, [chief_queue_runner])

        # Begin training
        logger.info("Starting training...")
        time_begin = time.time()
        logger.info(f"Training begins @ {time_begin}")

        # Load MNIST data
        logger.info("Loading MNIST data...")
        (x_train, y_train), (x_val, y_val) = mnist.load_data()
        x_train = (
            x_train.reshape(-1, IMAGE_PIXELS * IMAGE_PIXELS).astype("float32") / 255.0
        )
        x_val = x_val.reshape(-1, IMAGE_PIXELS * IMAGE_PIXELS).astype("float32") / 255.0
        logger.info("MNIST data loaded.")

        num_examples = x_train.shape[0]
        batch_size = FLAGS.batch_size
        steps_per_epoch = num_examples // batch_size
        logger.info(f"Number of examples: {num_examples}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Steps per epoch: {steps_per_epoch}")

        local_step = 0
        step = 0

        while step < FLAGS.train_steps:
            for _ in range(steps_per_epoch):
                batch_start = (local_step * batch_size) % num_examples
                batch_end = batch_start + batch_size
                batch_xs = x_train[batch_start:batch_end]
                batch_ys = y_train[batch_start:batch_end]
                train_feed = {x: batch_xs, y_: batch_ys}

                _, step = sess.run([train_step, global_step], feed_dict=train_feed)
                local_step += 1

                now = time.time()
                logger.info(
                    f"{now}: Worker {FLAGS.task_index}: training step {local_step} done (global step: {step})"
                )

                if step >= FLAGS.train_steps:
                    break

        time_end = time.time()
        logger.info(f"Training ends @ {time_end}")
        training_time = time_end - time_begin
        logger.info(f"Training elapsed time: {training_time} seconds")

        # Evaluate the model
        logger.info("Evaluating the model...")
        val_feed = {x: x_val, y_: y_val}
        val_loss = sess.run(cross_entropy, feed_dict=val_feed)
        logger.info(
            f"After {FLAGS.train_steps} training steps, validation cross entropy = {val_loss}"
        )


if __name__ == "__main__":
    tf.app.run()
