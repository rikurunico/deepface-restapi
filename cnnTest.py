import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import time

# Load dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalisasi pixel values ke range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Bangun model CNN sederhana
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)  # 10 kelas output untuk CIFAR-10
    ])
    return model

# Fungsi untuk melatih model pada perangkat tertentu
def train_on_device(device_name, epochs=5):
    with tf.device(device_name):
        # Buat model
        model = create_model()
        
        # Compile model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        
        # Catat waktu mulai
        start_time = time.time()
        
        # Latih model
        model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
        
        # Hitung waktu eksekusi
        elapsed_time = time.time() - start_time
        print(f"\nWaktu pelatihan pada {device_name}: {elapsed_time:.2f} detik")

# Jalankan pada CPU
print("Melatih model pada CPU...")
train_on_device("/CPU:0", epochs=5)

# Jalankan pada GPU
print("\nMelatih model pada GPU...")
train_on_device("/GPU:0", epochs=5)