import torch
import Constants as cts
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import datasets, transforms
from data.DataLoader import load_dataset, DataTransformationMNIST


def main():
    # Create device to perform computations in GPU (if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load pytorch model
    path = "./results/SVHN_250_unbalanced_3_downsampling_50/best_model/final_model_.pt"
    model = torch.load(path)
    model.to(device)

    # Data Loader
    if cts.SECOND_DATASET[0] == 'MNIST':
        test_loader_mnist = datasets.MNIST(cts.SECOND_DATASET[5], train=False, download=True)
        test_loader_aux = DataTransformationMNIST(cts.SECOND_DATASET[5], np.arange(len(test_loader_mnist)),
                                                  transform=transforms.ToTensor(), train=False)
        test_loader = DataLoader(test_loader_aux, sampler=SequentialSampler(test_loader_aux),
                                 batch_size=50, num_workers=0,
                                 drop_last=True, pin_memory=True)
        labels = np.array(test_loader_mnist.targets)

    else:
        # Obtaining test data from SVHN and labels
        _, test_data = load_dataset(cts.SVHN[0])
        labels = np.array(test_data.labels)
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

    # Compare the test labels to the predictions
    match = (labels == pred) * 1
    # Compute accuracy by comparing correct and wrong predictions
    accuracy = np.mean(match)

    print("Accuracy is: " + str(accuracy))
    accuracy = []
    for i in range(cts.DATASET[4]):
        tmp_indeces = np.where(labels == i)[0]
        if len(tmp_indeces) < 1:
            accuracy.append(0.0)
        else:
            match = (labels[tmp_indeces] == np.array(pred)[tmp_indeces]) * 1
            accuracy.append(np.mean(match))

    accuracy_per_class = np.array(accuracy)
    print("Accuracy per class is: ")
    print(accuracy_per_class)
    conf_matrix = confusion_matrix(labels, np.array(pred), normalize='true')
    conf_matrix = np.round(conf_matrix * 100, 2)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    plt.savefig("./results/SVHN_250_unbalanced_3_downsampling_50.png")
    plt.show()


if __name__ == "__main__":
    main()
