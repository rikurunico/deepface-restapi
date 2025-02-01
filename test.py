import tensorflow as tf
import time

# Ukuran matriks besar
size = 10000

# Buat matriks acak
A = tf.random.normal([size, size])
B = tf.random.normal([size, size])


# Fungsi untuk mengukur waktu eksekusi
def run_on_device(device_name):
    with tf.device(device_name):
        start_time = time.time()
        C = tf.matmul(A, B)
        elapsed_time = time.time() - start_time
        print(f"Waktu eksekusi pada {device_name}: {elapsed_time:.4f} detik")


# Jalankan pada CPU
run_on_device("/CPU:0")

# Jalankan pada GPU
run_on_device("/GPU:0")
