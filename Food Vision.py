import tensorflow_datasets as tfds
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from helper_functions import create_tensorboard_callback, plot_loss_curves, compare_historys

dataset_list = tfds.list_builders()  # Get all avaible datasets in TFDS
print("food101" in dataset_list)  # Is our target dataset is in the list

# Load in the data
(train_data, test_data), ds_info = tfds.load(name="food101",
                                             split=["train", "validation"],
                                             data_dir=r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision",
                                             # Path of where you want to download the dataset
                                             shuffle_files=True,
                                             as_supervised=True,  # Data gets returned in tuple format (data, label)
                                             with_info=True,
                                             download=False)  # Download=False if you load it before and want to run it again

sys.path.append(r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision")

# Features of Food101
print(ds_info.features)

# Get the class names
class_names = ds_info.features["label"].names
print(class_names[:10])

# Take one sample of train data
train_one_sample = train_data.take(1)
print("One sample: ", train_one_sample)

# Output info about training sample
for image, label in train_one_sample:
    print(f"""
  Image shape: {image.shape}
  Image dtype: {image.dtype}
  Target class from Food101 (tensor form): {label}
  Class name (str form): {class_names[label.numpy()]}
        """)

# What are the min and max values?
tf.reduce_min(image), tf.reduce_max(image)

# Plot an image tensor
plt.imshow(image)
plt.title(class_names[label.numpy()])  # add title to image by indexing on class_names list
plt.axis(False);


# Because of we shuffle_files=True it will give randomly

# Make a function for preprocessing images
def preprocess_img(image, label, img_shape=224):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]
    """
    image = tf.image.resize(image, [img_shape, img_shape])  # reshape to img_shape
    return tf.cast(image, tf.float32), label  # return (float32_image, label) tuple


# Preprocess a single sample image and check the outputs
preprocessed_img = preprocess_img(image, label)[0]
print(f"Image before preprocessing:\n {image[:2]}...,\nShape: {image.shape},\nDatatype: {image.dtype}\n")
print(
    f"Image after preprocessing:\n {preprocessed_img[:2]}...,\nShape: {preprocessed_img.shape},\nDatatype: {preprocessed_img.dtype}")

# Map preprocessing function to training data (and paralellize)
train_data = train_data.map(map_func=preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Shuffle train_data and turn it into batches and prefetch it (load it faster)
train_data = train_data.shuffle(buffer_size=1000).batch(batch_size=32).prefetch(buffer_size=tf.data.AUTOTUNE)

# Map prepreprocessing function to test data
test_data = test_data.map(preprocess_img, num_parallel_calls=tf.data.AUTOTUNE)
# Turn test data into batches (don't need to shuffle)
test_data = test_data.batch(32).prefetch(tf.data.AUTOTUNE)

print(train_data, test_data)

# Create ModelCheckpoint callback to save model's progress
checkpoint_path = r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision/cp.ckpt"  # saving weights requires ".ckpt" extension
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      monitor="val_accuracy",
                                                      # save the model weights with best validation accuracy
                                                      save_best_only=True,  # only save the best weights
                                                      save_weights_only=True,
                                                      # only save model weights (not whole model)
                                                      verbose=0)  # don't print out whether or not model is being saved

# Turn on mixed precision training
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy(policy="mixed_float16")  # set global policy to mixed precision

mixed_precision.global_policy()  # should output "mixed_float16" (if your GPU is compatible with mixed precision)

# Base Model
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

# Functional Model
inputs = tf.keras.layers.Input(shape=(224, 224, 3),
                               name="input_layer")

x = tf.keras.layers.Rescaling(1. / 255)(inputs)
x = base_model(inputs, training=False)(x)  # Set base_model to inference mode only
x = tf.keras.layers.GlobalAveragePooling2D(name="pooling_layer")(x)
x = tf.keras.layers.Dense(len(class_names))(x)
# Separate activation of output layer so we can output float32 activations
outputs = tf.keras.layers.Activation("softmax", dtype=tf.float32, name="softmax_float32")(x)
model = tf.keras.Model(inputs, outputs)

# Compile the model
model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

model.summary()

# Turn off all warnings except for errors
tf.get_logger().setLevel('ERROR')

history_feature_extraction = model.fit(train_data,
                                       epochs=3,
                                       validation_data=test_data,
                                       callbacks=[model_checkpoint, create_tensorboard_callback("logs",
                                                                                                "efficientnetb0_101_classes_all_data_feature_extract")])

# Evaluate model (unsaved version) on whole test dataset
results_feature_extract_model = model.evaluate(test_data)
print(results_feature_extract_model)

plot_loss_curves(history_feature_extraction)

# Saving Model

# # Create save path to drive
save_dir = r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision\07_efficientnetb0_feature_extract_model_mixed_precision"
# # os.makedirs(save_dir) # Make directory if it doesn't exist

# Save model
model.save(save_dir)

# Load model
loaded_saved_model = tf.keras.models.load_model(save_dir)
results_loaded_saved_model = loaded_saved_model.evaluate(test_data)
print(results_loaded_saved_model)

loaded_saved_model.summary()

# Trying Fine Tuning

# Unfreeze all the layers
loaded_saved_model.trainable = True

# Refreeze every layer except last 10
for layer in loaded_saved_model.layers[1].layers[:-10]:
    layer.trainable = False
    print(layer.name, layer.trainable, layer.dtype, layer.dtype_policy)

# Setup EarlyStopping callback to stop training if model's val_loss doesn't improve for 3 epochs
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",  # watch the val loss metric
                                                  patience=3)  # if val loss decreases for 3 epochs in a row, stop training

# Create ModelCheckpoint callback to save best model during fine-tuning
checkpoint_path = r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                      save_best_only=True,
                                                      monitor="val_loss")

# Creating learning rate reduction callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                 factor=0.2,  # multiply the learning rate by 0.2 (reduce by 5x)
                                                 patience=2,
                                                 verbose=1,  # print out when learning rate goes down
                                                 min_lr=1e-7)

# Recompile
loaded_saved_model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                           metrics=["accuracy"])

for layer in loaded_saved_model.layers:
    print(layer.name, layer.trainable)

for layer_number, layer in enumerate(loaded_saved_model.layers[1].layers):
    print(layer_number, layer.name, layer.trainable)

history_fine_tuning = loaded_saved_model.fit(train_data,
                                             epochs=100,
                                             validation_data=test_data,
                                             callbacks=[create_tensorboard_callback("logs",
                                                                                    "efficientb0_101_classes_all_data_fine_tuning"),
                                                        # track the model training logs
                                                        model_checkpoint,  # save only the best model during training
                                                        early_stopping,  # stop model after X epochs of no improvements
                                                        reduce_lr],
                                             initial_epoch=3)

fine_tuning_results = model.evaluate(test_data)

compare_historys(history_feature_extraction, history_fine_tuning)

loaded_saved_model.save(
    r"C:\Users\yazo_\OneDrive\Masaüstü\Tensorflow Projects\Food Vision\07_efficientnetb0_fine_tuning_model_mixed_precision")
