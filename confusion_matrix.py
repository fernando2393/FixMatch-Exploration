import torch
from data.data_loader import load_dataset
from Constants import SVHN
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns

# Create device to perform computations in GPU (if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pytorch model
path = "./results/SVHN_250_unbalanced_3_downsampling_50/best_model/final_model_.pt"
model = torch.load(path)
model.to(device)

# Obtaining test data from SVHN and labels
_, test_data = load_dataset(SVHN[0])
labels = np.array(test_data.labels)

# Data Loader
test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                         batch_size=64, num_workers=0,
                         pin_memory=True)

pred = []
for batch_idx, img_batch in enumerate(test_loader):
    # Define batch images and labels
    inputs, targets = img_batch

    # Evaluate method for the ema
    logits = model(inputs.to(device))[0]
    pred.extend(torch.argmax(logits, 1).cpu().numpy())

conf_matrix = confusion_matrix(labels, np.array(pred), normalize='true')
conf_matrix = np.round(conf_matrix * 100, 2)
sns.heatmap(conf_matrix, annot=True, cmap='Blues')
plt.savefig("./results/all_balanced_confusion_matrix.png")
plt.show()




