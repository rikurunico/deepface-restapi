import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.set_logical_device_configuration(gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=630)] * 12)
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    
logical_gpus = tf.config.list_logical_devices('GPU')
print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")