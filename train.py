import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.functional import cross_entropy
import torch.optim as optim


##### TRAINING #####
def train_fixmatch(model, device, labeled_train_data, unlabeled_train_data, learning_rate, B, unlabeled_batch_size, threshold):
    # Compute loss for labeled data (mean loss of images)
    supervised_loss = supervised_train(model, device, labeled_train_data, B)

    # Compute loss of unlabeled_data (mean loss of images)
    unsupervised_loss = unsupervised_train(model, device, unlabeled_train_data, unlabeled_batch_size, threshold)

    # Compute the total loss (SSL loss)
    semi_supervised_loss = supervised_loss + learning_rate * unsupervised_loss

    return semi_supervised_loss, supervised_loss, unsupervised_loss

def supervised_train(model, device, labeled_train_data, B):
    loss = []
    for batch_idx, img_batch in enumerate(labeled_train_data):
        # Define batch images and labels
        inputs, targets = img_batch

        # Make predictions
        predictions = model(inputs.to(device))

        # Compute loss of batch
        loss.extend(cross_entropy(predictions, targets.to(device), reduction='mean'))

    # Average over the labeled batch
    supervised_loss = sum(loss) / B
    return supervised_loss

def unsupervised_train(model, device, unlabeled_train_data, unlabeled_batch_size, threshold):
    loss = []
    for batch_idx, img_batch in enumerate(unlabeled_train_data):
        # Define batch images (weakly and strongly augmented) and not assing the target labels
        weakly_augment_inputs, strongly_augment_inputs, _ = img_batch

        # Assign Pseudo-labels and mask them based on threshold 
        pseudo_labels, masked_indeces = pseudo_labeling(model, weakly_augment_inputs.to(device), threshold)

        # Compute predictions for strongly augmented images
        strongly_predictions = model(strongly_augment_inputs.to(device))

        # Compute loss between pseudo-labeled images and strongly augmented images
        masked_loss = cross_entropy(strongly_predictions, pseudo_labels, reduction='mean')[masked_indeces]
        loss.extend(masked_loss)
        
    # Average over the unlabeled batch
    unsupervised_loss = sum(loss) / unlabeled_batch_size
    return unsupervised_loss

def pseudo_labeling(model, weakly_augment_inputs, threshold):
    all_probs = []
    # Detach the linear prediction 
    logits = model(weakly_augment_inputs.detach())

    # Compute pseudo probabilities
    probs = torch.softmax(logits, dim=1)

    # One hot encode pseudo labels and define mask for those that surpassed the threshold
    scores, pseudo_labels = torch.max(probs, dim=1)
    masked_indeces = scores >= threshold
    return pseudo_labels, masked_indeces

##### TESTING #####
def test_fixmatch(ema, model, test_data, B):
    # Compute accuract for the model and ema
    acc_model_tmp = [] # Creating a list might be interesting if you decide to check the performance of each batch
    acc_ema_tmp = []
    for batch_idx, img_batch in enumerate(test_data):
        # Define batch images and labels
        inputs, targets = img_batch
        
        # Evalutate method for the model 
        model.eval()
        logits = model(inputs)
        acc_model_tmp.extend(evaluate(logits, targets))

        # Evalutate method for the ema
        ema.copy_to(model.parameters())
        logits = model(inputs)
        acc_ema_tmp.extend(evaluate(logits, targets))

    # Compute the accuracy average over the batches (size B) 
    acc_model = sum(acc_model_tmp) / B
    acc_ema = sum(acc_ema_tmp) / B

    return acc_model, acc_ema

def evaluate(logits, targets):
    # Compute the scores
    scores = torch.softmax(logits, dim=1)

    # Return the predictions
    _, preds = torch.max(scores, dim=1)
    
    # Compare the test labels to the predictions
    match = (targets == preds)

    # Compute accuracy by comparing correct and wrong predictions
    accuracy = torch.mean(match)

    return accuracy