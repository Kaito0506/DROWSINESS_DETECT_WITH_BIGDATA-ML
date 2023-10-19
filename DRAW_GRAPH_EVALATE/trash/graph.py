import matplotlib.pyplot as plt

# Read data from the file
data = []
data_path = "/home/ubuntu/CLUSTER_TRAIN/output/ViT/training_history.txt"
with open(data_path, 'r') as file:
    for line in file:
        data.append(line.strip().split())

# Separate the data into different lists
epochs = [int(row[0]) for row in data[1:]]
train_acc = [float(row[1]) for row in data[1:]]
val_acc = [float(row[2]) for row in data[1:]]
train_loss = [float(row[3]) for row in data[1:]]
val_loss = [float(row[4]) for row in data[1:]]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,10))

# Subplot 1: Accuracy
ax1.plot(epochs, train_acc, label='Train Accuracy')
ax1.plot(epochs, val_acc, label='Validation Accuracy')
ax1.set_title('Training and Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()

# Subplot 2: Loss
ax2.plot(epochs, train_loss, label='Train Loss')
ax2.plot(epochs, val_loss, label='Validation Loss')
ax2.set_title('Training and Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
plt.subplots_adjust(hspace=0.5, wspace=0.1)
# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()
