from re import X
import numpy as np
import random
import json


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import tokenize , stem, bag_of_words
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = [] # all the words patterns
tags = [] # different type of intents or tags
xy = [] # patterns and tags

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)  # use extend insted of append because w is an array so don't want to append an array in a array so extend.
        xy.append((w, tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
# print(tags)


#Traning Data

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) 

X_train = np.array(X_train)
y_train = np.array(y_train)

#Hyperparameter 
num_epochs = 1000
batch_size = 10
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 20
output_size = len(tags)
print(input_size, output_size)
# print(input_size, len(all_words))
# print(output_size, tags)


class ChatDataset(Dataset):
    def __init__(self) -> None:
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    #datasets[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# data
dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size, 
    shuffle=True,
    num_workers=0
    )


device = torch.device('cpu')

# MODEL
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# LOSS and OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # FORWARD PASS
        outputs = model(words)
        loss = criterion(outputs, labels)

        # BACKWARD and Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 100 ==0:
        print(f'epoch: {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f"final loss, loss= {loss.item():.4f}")


# SAVING THE MODEL AND ITS DATA
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "fastAidData.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")