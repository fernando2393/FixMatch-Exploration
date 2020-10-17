import torch
from torch.nn import CrossEntropyLoss


# -----TRAINING----- #
def train_fixmatch(model, device, labeled_train_data, unlabeled_train_data, lambda_unsupervised, B,
                   unlabeled_batch_size, threshold):
    # Compute loss for labeled data (mean loss of images)
    supervised_loss = supervised_train(model, device, labeled_train_data, B)

    # Compute loss of unlabeled_data (mean loss of images)
    unsupervised_loss = unsupervised_train(model, device, unlabeled_train_data,
                                                                   unlabeled_batch_size, threshold)

    # Compute the total loss (SSL loss)
    semi_supervised_loss = supervised_loss + lambda_unsupervised * unsupervised_loss

    return semi_supervised_loss, supervised_loss, unsupervised_loss


def supervised_train(model, device, labeled_train_data, B):
    loss = 0
    n_batches = 0
    for batch_idx, img_batch in enumerate(labeled_train_data):
        # Define batch images and labels
        inputs, targets = img_batch

        # Make predictions
        predictions = model(inputs.to(device))[0]  # Item 0 -> Output. Items 1, 2, 3 -> Attention

        # Compute loss of batch
        criterion = CrossEntropyLoss()
        loss_tmp = criterion(predictions, targets.long().to(device))
        # loss.extend(CrossEntropyLoss(predictions, targets.to(device), reduction='mean'))
        loss += loss_tmp
        n_batches += 1


    # Average over the labeled batch
    supervised_loss = loss / n_batches
    return supervised_loss


def unsupervised_train(model, device, unlabeled_train_data, unlabeled_batch_size, threshold):
    loss = 0
    n_batches = 0
    for batch_idx, (image_batch, _) in enumerate(unlabeled_train_data):
        # Define batch images (weakly and strongly augmented) and not assing the target labels
        weakly_augment_inputs, strongly_augment_inputs = image_batch

        # Assign Pseudo-labels and mask them based on threshold 
        pseudo_labels, masked_indeces = pseudo_labeling(model, weakly_augment_inputs.to(device), threshold)

        if True not in masked_indeces:
            loss += torch.tensor(0.0).to(device) # Append a 0 if no image surpassed the threshold
        else:
            # Compute predictions for strongly augmented images
            strongly_predictions = model(strongly_augment_inputs.to(device))[0]
            # Compute loss of batch
            criterion = CrossEntropyLoss()
            loss_tmp = criterion(strongly_predictions[masked_indeces], pseudo_labels[masked_indeces].to(device))
            loss += loss_tmp
        n_batches += 1

    # Average over the unlabeled batch
    unsupervised_loss = loss / n_batches
    return unsupervised_loss


def pseudo_labeling(model, weakly_augment_inputs, threshold):
    # Detach the linear prediction 
    logits = model(weakly_augment_inputs.detach())[0]

    # Compute pseudo probabilities
    probs = torch.softmax(logits, dim=1)

    # One hot encode pseudo labels and define mask for those that surpassed the threshold
    scores, pseudo_labels = torch.max(probs, dim=1)
    masked_indeces = (scores >= threshold)
    return pseudo_labels, masked_indeces


# -----TESTING----- #
def test_fixmatch(ema, model, test_data, B, device):
    # Compute accuract for the model and ema
    acc_model_tmp = 0  # Creating a list might be interesting if you decide to check the performance of each batch
    acc_ema_tmp = 0
    # Evalutate method for the model
    model.eval()
    with torch.no_grad():
        n_batches = 0
        for batch_idx, img_batch in enumerate(test_data):
            # Define batch images and labels
            inputs, targets = img_batch

            logits = model(inputs.to(device))[0]
            acc_model_tmp += evaluate(logits, targets.to(device))

            # Evalutate method for the ema
            ema.copy_to(model.parameters())
            logits = model(inputs.to(device))[0]
            acc_ema_tmp += evaluate(logits, targets.to(device))
            n_batches += 1

    # Compute the accuracy average over the batches (size B) 
    acc_model = acc_model_tmp / n_batches
    acc_ema = acc_ema_tmp / n_batches

    return acc_model, acc_ema


def evaluate(logits, targets):
    # Compute the scores
    scores = torch.softmax(logits, dim=1)

    # Return the predictions
    _, preds = torch.max(scores, dim=1)

    # Compare the test labels to the predictions
    match = (targets == preds) * 1

    # Compute accuracy by comparing correct and wrong predictions
    accuracy = torch.mean(match.float())

    return accuracy
