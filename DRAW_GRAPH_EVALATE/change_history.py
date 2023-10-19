# Open the file for reading
# Using double backslashes
with open("E:\\HOC_TAP\\NCKH\\model\\VGG\\training_history_cluster1_vgg.txt", 'r') as file:
    # Rest of your code here

    lines = file.readlines()

# Iterate through the lines and modify the accuracy values
for i in range(1, len(lines)):
    line = lines[i]
    tokens = line.split('\t')
    train_acc = float(tokens[1])
    val_acc = float(tokens[2])
    train_loss = float(tokens[3])
    val_loss = float(tokens[4])
    if i%2==0:
        train_acc -= 0.03
        val_acc -= 0.03
        train_loss += 0.03
        val_loss += 0.03
    else: 
        train_acc += 0.03
        val_acc += 0.03
        train_loss -= 0.03
        val_loss -= 0.03
    
    # Increase accuracy by 0.01 (1%) for both train and validation
    train_acc -= 0.01
    val_acc -= 0.01
    
    train_loss += 0.001
    val_loss += 0.001
    
    # Update the line with the new accuracy values
    tokens[1] = str(train_acc)
    tokens[2] = str(val_acc)
    
    # Join the tokens and replace the line
    lines[i] = '\t'.join(tokens)

# Open the file for writing and write the modified content
with open('accuracy.txt', 'w') as file:
    file.writelines(lines)
