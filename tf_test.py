import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("GPU devices:", tf.config.list_physical_devices('GPU'))

if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU detected successfully!")