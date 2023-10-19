
import matplotlib.pyplot as plt
import numpy as np
# Read data from the file
data = []
data_path = r"E:\HOC_TAP\NCKH\model\resnet\training_history_resnet_cluster1.txt"
with open(data_path, 'r') as file:
    for line in file:
        data.append(line.strip().split())

# Separate the data into different lists
epochs = [int(row[0]) for row in data[1:]]
train_acc = [float(row[1]) for row in data[1:]]
val_acc = [float(row[2]) for row in data[1:]]
train_loss = [float(row[3]) for row in data[1:]]
val_loss = [float(row[4]) for row in data[1:]]


# vgg  = ["local VGG16", "cluster 1 VGG", "cluster 2 VGG"]
# dense = ["local Densenet121", "cluster 1 Densenet121", "cluster 2 Densenet121"]
# lstm = ["local LSTM", "cluster 1 LSTM", "cluster 2 LSTM"]
# vit = ["local ViT", "cluster 1 ViT", "cluster 2 ViT"]
# resnet = ["local Resnet152", "cluster 1 Resnet152", "cluster 2 Resnet152"]

# initalize an aaray with the name scenario and 15 columns inside "1" "2".....
scenario = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
name = scenario[1]
gs = plt.GridSpec(2, 1, width_ratios=[1], height_ratios=[1, 1])  # 2 rows, 1 column
plt.figure(figsize=(4,5))
# First subplot for x_data and y_data
plt.subplot(gs[0])  # Use the first row
plt.plot(train_acc)
plt.plot(val_acc)
plt.title("Sencario " + name + ' Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
# x_ticks = np.arange(0, 501, 50)  # You can adjust this interval as needed
# plt.xticks(x_ticks)
# # Customize y ticks with a minimum interval of 0.1
# y_ticks = np.arange(0.4, 1.1, 0.1)  # You can adjust this interval as needed
# plt.yticks(y_ticks)


# Plot training and validation loss
plt.subplot(gs[1])  # Use the second row
plt.plot(train_loss)
plt.plot(val_loss)
plt.title("Scenario " + name + ' Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
# x_ticks = np.arange(0, 501, 50)  # You can adjust this interval as needed
# plt.xticks(x_ticks)
# # # Customize y ticks with a minimum interval of 0.1
# y_ticks = np.arange(0, 3, 0.5)  # You can adjust this interval as needed
# plt.yticks(y_ticks)
plt.subplots_adjust(hspace=0.5, wspace=0.1)
plt.show()