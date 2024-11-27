import tensorflow as tf
import os
import json
import logging
import time

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    logger.info(f"GPUs detected: {gpus}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        logger.debug(f"Enabled memory growth for GPU: {gpu}")
else:
    logger.error(
        "No GPUs detected. Ensure ROCm TensorFlow is installed and GPU is configured."
    )
    raise RuntimeError("No GPU available.")

# Load TensorFlow config from environment variable
tf_config_str = os.environ.get("TF_CONFIG", "{}")
logger.info(f"TF_CONFIG: {tf_config_str}")

try:
    tf_config = json.loads(tf_config_str)
except json.JSONDecodeError as e:
    logger.error(f"Failed to parse TF_CONFIG: {e}")
    raise

task_type = tf_config.get("task", {}).get("type", "")
task_index = tf_config.get("task", {}).get("index", 0)
logger.info(f"Task Type: {task_type}, Task Index: {task_index}")

# Strategy for multi-worker distributed training
logger.info("Initializing MultiWorkerMirroredStrategy...")
strategy = tf.distribute.MultiWorkerMirroredStrategy()
logger.info(f"Strategy initialized: {strategy}")

global_batch_size = 128
logger.info(f"Global batch size: {global_batch_size}")


def preprocess_dataset(x, y):
    logger.debug("Preprocessing dataset...")
    x = tf.cast(x, tf.float32) / 255.0
    y = tf.cast(y, tf.int64)
    return x, y


def create_dataset(input_context, training=True):
    logger.info(
        f"Creating {'training' if training else 'testing'} dataset with input context: {input_context}"
    )
    batch_size = input_context.get_per_replica_batch_size(global_batch_size)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if training:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    dataset = dataset.map(preprocess_dataset)
    dataset = dataset.shard(
        input_context.num_input_pipelines, input_context.input_pipeline_id
    )
    if training:
        dataset = dataset.shuffle(10000).repeat()
    dataset = dataset.batch(batch_size)
    logger.debug("Dataset created.")
    return dataset


logger.info("Distributing datasets...")
train_dist_dataset = strategy.distribute_datasets_from_function(
    lambda input_context: create_dataset(input_context, training=True)
)
test_dist_dataset = strategy.distribute_datasets_from_function(
    lambda input_context: create_dataset(input_context, training=False)
)

log_dir = f"logs/worker_{task_index}"
checkpoint_dir = f"ckpt/worker_{task_index}.keras"
logger.info(f"Log directory: {log_dir}, Checkpoint directory: {checkpoint_dir}")

# Model creation
logger.info("Building the model...")
with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),  # Input layer
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10),  # Output layer
        ]
    )
    logger.info("Model built.")
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    logger.debug(f"Optimizer configured: {optimizer}")
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    logger.debug("Loss object configured.")


@tf.function
def distributed_train_step():
    def step_fn(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            per_example_loss = loss_object(y, predictions)
            per_replica_loss = tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_batch_size
            )
        gradients = tape.gradient(per_replica_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return per_replica_loss

    return strategy.run(step_fn, args=(next(train_iterator),))


@tf.function
def distributed_test_step():
    def step_fn(inputs):
        x, y = inputs
        predictions = model(x, training=False)
        per_example_loss = loss_object(y, predictions)
        per_replica_loss = tf.nn.compute_average_loss(
            per_example_loss, global_batch_size=global_batch_size
        )
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(predictions, axis=-1), y), tf.float32)
        )
        return per_replica_loss, accuracy

    return strategy.run(step_fn, args=(next(test_iterator),))


train_iterator = iter(train_dist_dataset)
test_iterator = iter(test_dist_dataset)

# Training loop
for epoch in range(2):
    logger.info(f"Starting epoch {epoch+1}...")
    start_time = time.time()
    total_loss = 0.0

    for step in range(200):
        loss_value = distributed_train_step()
        loss_value_mean = strategy.reduce(
            tf.distribute.ReduceOp.SUM, loss_value, axis=None
        )
        total_loss += loss_value_mean.numpy()
        if step % 10 == 0:
            logger.info(f"Step {step}, Loss: {loss_value_mean.numpy()}")

    avg_loss = total_loss / 200
    epoch_time = time.time() - start_time
    logger.info(
        f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}, Time taken: {epoch_time:.2f}s"
    )

    # Save model checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_weights(f"{checkpoint_dir}/weights_epoch_{epoch+1}.weights.h5")
    logger.info(f"Model checkpoint saved for epoch {epoch+1}.")

    # Evaluation
    logger.info(f"Evaluating at epoch {epoch+1}...")
    total_test_loss, total_test_accuracy = 0.0, 0.0
    for _ in range(20):
        test_loss, test_accuracy = distributed_test_step()
        total_test_loss += strategy.reduce(
            tf.distribute.ReduceOp.SUM, test_loss, axis=None
        ).numpy()
        total_test_accuracy += strategy.reduce(
            tf.distribute.ReduceOp.MEAN, test_accuracy, axis=None
        ).numpy()

    logger.info(
        f"Test Loss: {total_test_loss / 20:.4f}, Test Accuracy: {total_test_accuracy / 20:.4f}"
    )

# Save and push model checkpoints to a remote repository or storage service, such as HuggingFace Hub for version control and sharing, object storage like AWS S3, Google Cloud Storage, or Azure Blob Storage, or a custom file server using protocols like FTP or WebDAV.
