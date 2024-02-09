import GPUtil
import time

while True:
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print("GPU {}: {:.2f}% Utilization".format(gpu.id, gpu.load * 100))
    time.sleep(1)  # Adjust the sleep time as needed
