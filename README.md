# FixMatch & SSL Exploration
Final project of the KTH DD2412 - Deep Learning, Advanced course.

The scope of this project is to replicate [FixMatch](https://arxiv.org/pdf/2001.07685.pdf),
a method developed by Google researchers in order to perform semi-supervised learning.

The tasks we performed are the followings:
1. Replication of the method using PyTorch and training on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [SVHN](http://ufldl.stanford.edu/housenumbers/) datasets.
2. Unbalance the supervised classes number of samples when training.
3. Varying the labeled-unlabeled data ratio.
4. Employing the merge of different datasets of the same nature as unlabeled data, SVHN and [MNIST](http://yann.lecun.com/exdb/mnist/) in this case.

The main outcomes obtained in each task can be seen below. In order to get a deeper insight of
our findings, go to [FixMatch & SLL Exploration](fixmatch_and_ssl_exploration.pdf).
* Task 1:

| Dataset        | Labeled Samples  | Test Accuracy     |
| -------------- |:---------------: | -----------------:|
| CIFAR-10       | 4000             | 90.70%            |
| CIFAR-10       | 250              | 85.44%            |
| CIFAR-10       | 40               | 64.44%            |
| SVHN           | 250              | 95.29%            |

| <img alt="Accuracy CIFAR-10 40 labels" src="/Results/CIFAR-10_40/Accuracy40.png" width="250"/> | <img alt="Accuracy CIFAR-10 250 labels" src="/Results/CIFAR-10_250/Accuracy250.png" width="250"/> |
|:--:|:--:|
| *Accuracy CIFAR-10 40 labels.* | *Accuracy CIFAR-10 250 labels.*|

| <img alt="Accuracy CIFAR-10 4000 labels" src="/Results/CIFAR-10_4000/Accuracy4000.png" width="250"/> | <img alt="Accuracy SVHN 250 labels" src="/Results/SVHN_250/Accuracy250.png" width="250"/> |
|:--:|:--:|
| *Accuracy CIFAR-10 4000 labels.* | *Accuracy SVHN 250 labels.*|


* Task 2:

| Dataset        | Scaling Factor  | Test Accuracy Class 3 | Test Accuracy Class 5 |
| -------------- |:---------------: | :------------------: | :-------------------: |
| SVHN           | 1                | 92.44%               | 95.18%
| SVHN           | 0.5              | 91.29%               | 96.35%
| SVHN           | 1.5              | 95.62%               | 97.06%

| <img alt="Confusion matrix SVHN class 3 not scaled" src="/Results/SVHN_Unbalanced_Class/all_balanced_confusion_matrix.png" width="200"/> | <img alt="Confusion matrix SVHN class 3 downscaled 50%" src="/Results/SVHN_Unbalanced_Class/downsampling_confusion_matrix.png" width="200"/> | <img alt="Confusion matrix SVHN class 3 upscaled 50%" src="/Results/SVHN_Unbalanced_Class/oversampling_confusion_matrix.png" width="200"/> |
|:--:|:--:|:--:|
| *Confusion matrix SVHN class 3 not scaled.* | *Confusion matrix SVHN class 3 downscaled 50%.*| *Confusion matrix SVHN class 3 upscaled 50%* |

* Task 3:

| Dataset        | Percentage of Total Unlabeled Data  | Test Accuracy     |
| -------------- |:----------------------------------: | -----------------:|
| CIFAR-10       | 25%                                 | 49.00%            |
| CIFAR-10       | 50%                                 | 51.23%            |
| CIFAR-10       | 75%                                 | 57.93%            |
| CIFAR-10       | 100%                                | 64.44%            |

| <img alt="Accuracy CIFAR-10 25% unlabeled data" src="/Results/CIFAR-10_40_0.25/Accuracy40_0.25.png" width="250"/> | <img alt="Accuracy CIFAR-10 50% unlabeled data" src="/Results/CIFAR-10_40_0.5/Accuracy40_0.5.png" width="250"/> |
|:--:|:--:|
| *Accuracy CIFAR-10 25% unlabeled data.* | *Accuracy CIFAR-10 50% unlabeled data.*|

| <img alt="Accuracy CIFAR-10 75% unlabeled data" src="/Results/CIFAR-10_40_0.75/Accuracy40_0.75.png" width="250"/> | <img alt="Accuracy CIFAR-10 100% unlabeled data" src="/Results/CIFAR-10_40/Accuracy40.png" width="250"/> |
|:--:|:--:|
| *Accuracy CIFAR-10 75% unlabeled data.* | *Accuracy CIFAR-10 100% unlabeled data.*|

* Task 4:

| Labeled Data Dataset | Unlabeled Data Datasets | Test Dataset | Test Accuracy |
| -------------------- |:----------------------: | :----------: | :-----------: |
| SVHN                 | SVHN                    | SVHN         | 95.29%
| SVHN                 | SVHN                    | MNIST        | 73.01%
| SVHN                 | SVHN and MNIST          | SVHN         | 96.51%
| SVHN                 | SVHN and MNIST          | MNIST        | 86.04%

| <img alt="Confusion matrix SVHN merged unlabeled data" src="/Results/SVHN_250_MNIST/confusion_matrix_SVHN.png" width="250"/> | <img alt="Confusion matrix MNIST merged unlabeled data" src="/Results/SVHN_250_MNIST/confusion_matrix_MNIST.png" width="250"/> |
|:--:|:--:|
| *Confusion matrix SVHN merged unlabeled data.* | *Confusion matrix MNIST merged unlabeled data.*|
