import torch
import numpy as np
from torch.nn import CrossEntropyLoss
import Constants as cts


# -----TRAINING----- #
def train_fixmatch(model, device, labeled_image_batch, labeled_targets_batch, unlabeled_image_batch,
                   unlabeled_batch_size, lambda_unsupervised, threshold):
    # Compute loss for labeled data (mean loss of images)
    supervised_loss = supervised_train(model, device, labeled_image_batch, labeled_targets_batch)

    # Compute loss of unlabeled_data (mean loss of images)
    unsupervised_loss, unsupervised_ratio = unsupervised_train(model, device, unlabeled_image_batch,
                                                               unlabeled_batch_size, threshold)

    # Compute the total loss (SSL loss)
    semi_supervised_loss = supervised_loss + lambda_unsupervised * unsupervised_loss

    return semi_supervised_loss, supervised_loss, unsupervised_loss, unsupervised_ratio


def supervised_train(model, device, inputs, targets):
    # Make predictions
    predictions = model(inputs.to(device))[0]  # Item 0 -> Output. Items 1, 2, 3 -> Attention

    # Compute loss of batch
    criterion = CrossEntropyLoss()
    supervised_loss = criterion(predictions, targets.long().to(device))

    # Free space
    del predictions

    return supervised_loss


def unsupervised_train(model, device, unlabeled_image_batch, unlabeled_batch_size, threshold):
    # Define batch images (weakly and strongly augmented) and not assing the target labels
    weakly_augment_inputs, strongly_augment_inputs = unlabeled_image_batch

    # Assign Pseudo-labels and mask them based on threshold
    pseudo_labels, masked_indeces = pseudo_labeling(model, weakly_augment_inputs.to(device), threshold)

    # Declare ratio
    unsupervised_ratio = 0

    if True not in masked_indeces:
        unsupervised_loss = torch.tensor(0.0)  # 0 if no image surpassed the threshold
    else:
        # Compute predictions for strongly augmented images
        augment_with_pseudolabels = strongly_augment_inputs[masked_indeces]
        strongly_predictions = model(augment_with_pseudolabels.to(device))[0]

        # Compute loss of batch
        criterion_unsupervised = CrossEntropyLoss(reduction='sum')
        unsupervised_loss = criterion_unsupervised(strongly_predictions,
                                                   pseudo_labels[masked_indeces].to(device)) / unlabeled_batch_size

        # Compute number of unsupervised images used
        unsupervised_ratio = sum(masked_indeces.tolist()) / len(masked_indeces.tolist())

    return unsupervised_loss, unsupervised_ratio


def pseudo_labeling(model, weakly_augment_inputs, threshold):
    # Set model to no grad so it won't change the gradients based on the pseudo-labels
    with torch.no_grad():
        logits = model(weakly_augment_inputs)[0]

        # Compute the probabilities
        probs = torch.softmax(logits.detach(), dim=1)

        # Free space
        del logits

        # One hot encode pseudo labels and define mask for those that surpassed the threshold
        scores, pseudo_labels = torch.max(probs, dim=1)
        masked_indeces = (scores >= threshold)
    return pseudo_labels, masked_indeces


# -----TESTING----- #
def test_fixmatch(ema, test_data, device, last=False):
    # Compute accuracy for the model and ema
    acc_ema_tmp = 0

    # Evalutate method for the model
    if not last:
        ema.eval()
    with torch.no_grad():
        n_batches = 0
        if cts.DATASET[0] != "SVHN":
            for batch_idx, img_batch in enumerate(test_data):
                # Define batch images and labels
                inputs, targets = img_batch

                # Evaluate method for the ema
                logits = ema(inputs.to(device))[0]
                acc_ema_tmp += evaluate(logits, targets.to(device))
                n_batches += 1

                # Free space
                del logits

        else:
            for batch_idx, img_batch in enumerate(test_data):
                # Define batch images and labels
                inputs, targets = img_batch

                # Evaluate method for the ema
                logits = ema(inputs.to(device))[0]
                acc_ema_tmp += evaluate(logits, targets, device)
                n_batches += 1

    # Compute the accuracy average over the batches (size B)
    acc_ema = acc_ema_tmp / n_batches

    return acc_ema


def evaluate(logits, targets, device=None):
    # Return the predictions
    _, preds = torch.max(logits, dim=1)

    if cts.DATASET[0] == "SVHN":
        accuracy = list()
        for i in range(cts.DATASET[4]):
            tmp_indeces = np.where(targets == i)[0]
            if len(tmp_indeces) < 1:
                accuracy.append(torch.tensor(0.0))
            else:
                match = (targets[tmp_indeces].to(device) == preds[tmp_indeces]) * 1
                accuracy.append(torch.mean(match.float()))

        return np.array(accuracy)

    # Compare the test labels to the predictions
    match = (targets == preds) * 1

    # Compute accuracy by comparing correct and wrong predictions
    accuracy = torch.mean(match.float())

    return accuracy
