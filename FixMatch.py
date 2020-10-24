import random
import torch
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from train import train_fixmatch, test_fixmatch
import CTAugment as ctaug
from data.data_loader import *
import torch.optim as optim
from exp_moving_avg import EMA
from WideResNet_PyTorch.src import WideResNet as wrn
import os
import Constants as cts


# -----SET RANDOMNESS----- #
def set_seed(seed=42):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)


# -----DEFINE FUNCTIONS----- #
def cyclic_learning_rate_with_warmup(warmup_steps, step, total_training_steps):
    # If you don't achieve the number of warmup steps, don't update

    def scheduler_function(step):
        if step < warmup_steps:
            return float(step) / float(warmup_steps)

        else:  # Once you surpass the number of warmup steps,
            # you should decay they learning rate close zero in a cosine manner
            x = np.cos(7. / 16. * np.pi * ((step - warmup_steps) / (total_training_steps - warmup_steps)))
            return x

    # Update learning rate scheduler
    return scheduler_function


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def plot_performance(title, x_label, y_label, x_data, y_data, color=None):
    plt.plot(x_data, y_data, label=title, c=color)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()


def main():
    set_seed(42)
    CB91_Blue = '#2CBDFE'
    CB91_Green = 'springgreen'
    CB91_Red = '#DA6F6F'
    n_labeled_data = 250  # We will train with 250 labeled data to avoid computing many times the CTAugment
    B = 64  # B from the paper, i.e. number of labeled examples per batch.
    mu = 7  # Hyperparam of Fixmatch determining the relative number of unlabeled examples w.r.t. B * mu
    unlabeled_batch_size = B * mu
    initial_learning_rate = 0.03  # Small learning rate, which with cyclic decay will tend to zero
    momentum = 0.9  # Momentum to access the Stochastic Gradient Descent
    nesterov_factor = True  # They found that the nesterov hyperparm wasn't necessary to achieve errors below 5%
    pseudo_label_threshold = 0.95  # Threshold to guarantee confidence on the model
    total_training_epochs = 2 ** 10  # Number of training epochs, without early stopping (assuming the model
    # expects to see 2^26 images during the whole training)
    initial_training_step = 0  # Start the training epoch from zero
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")  # Create device to perform computations in GPU (if available)
    ema_decay = 0.999
    weight_decay = 0.0005
    total_training_steps = 2 ** 20  # Number of training epochs, without early stopping (assuming the model

    # -----Define WideResNet Architecture-----#
    wrn_depth = 28
    wrn_width = 2
    strides = [1, 1, 2, 2]
    channels = 3  # Maybe this has to be changed in order to support grayscale

    # -----START MODEL----- #

    # Create Wide - ResNet based on the data set characteristics
    model = wrn.WideResNet(d=wrn_depth, k=wrn_width, n_classes=cts.DATASET[4], input_features=channels,
                           output_features=16, strides=strides)

    model.to(device)

    # Analyze the training process
    acc_ema = []
    supervised_loss_list = []
    unsupervised_loss_list = []
    semi_supervised_loss_list = []
    unsupervised_ratio_list = []

    # Query datasets
    # 'sample_proportion' has to go in between 0 and 1
    labeled_indeces, unlabeled_indeces, test_data = dataset_loader(cts.DATASET[0], num_labeled=n_labeled_data,
                                                                   balanced_split=True)

    # Reshape indeces to have the same number of batches
    n_unlabeled_images = len(unlabeled_indeces)  # CIFAR - 49750 unlabeled for 250 labeled
    n_complete_batches = (n_unlabeled_images // unlabeled_batch_size)  # Number of complete batches 111
    n_images_in_complete_batches = n_complete_batches * B  # 7104
    n_labeles_times = (n_images_in_complete_batches // n_labeled_data)  # 28
    reminder = (n_images_in_complete_batches % n_labeled_data) + B  # 104 + batch size
    labeled_indeces_extension = []
    labeled_indeces_extension.extend(labeled_indeces * n_labeles_times)
    labeled_indeces_extension.extend(labeled_indeces[:reminder])
    warmup_steps = 10 * (n_complete_batches + 1)  # Define number of warmup steps to avoid premature cyclic learning

    # Define Stochastic Gradient Descent
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=momentum, nesterov=nesterov_factor,
                          weight_decay=weight_decay)

    # Create scheduler that will take charge of warming up and performing learning rate decay
    # LambdaLR: Sets the learning rate of each parameter group to the initial lr times a given function.
    # (Pytorch documentation)
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=cyclic_learning_rate_with_warmup(warmup_steps,
                                                                                         initial_training_step,
                                                                                         total_training_steps))

    # Define CTA augmentation
    cta = ctaug.CTAugment(depth=2, t=0.8, ro=0.99)

    # Apply transformations
    labeled_dataset, unlabeled_dataset, train_label_cta = applyTransformations(cts.DATA_ROOT,
                                                                               labeled_indeces_extension,
                                                                               labeled_indeces_extension,
                                                                               unlabeled_indeces,
                                                                               cts.DATASET[1],
                                                                               cts.DATASET[2],
                                                                               cta)

    # Load datasets
    labeled_train_data = DataLoader(labeled_dataset, batch_size=B,
                                    sampler=RandomSampler(labeled_dataset),
                                    num_workers=16,
                                    drop_last=True,
                                    pin_memory=True)

    unlabeled_train_data = DataLoader(unlabeled_dataset, sampler=RandomSampler(unlabeled_dataset),
                                      batch_size=unlabeled_batch_size, num_workers=0,
                                      drop_last=True, pin_memory=True)

    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data),
                             batch_size=B, num_workers=16,
                             pin_memory=True)

    labeled_train_cta_data = DataLoader(train_label_cta, sampler=SequentialSampler(train_label_cta),
                                        batch_size=B,
                                        num_workers=0,
                                        drop_last=True,
                                        pin_memory=True)

    # Compute best accuracy
    best_acc = 0

    for epoch in tqdm(range(total_training_epochs)):
        print('TRAINING epoch', epoch + 1)

        # Declare lists of training
        semi_supervised_loss_list_tmp = []
        supervised_loss_list_tmp = []
        unsupervised_loss_list_tmp = []
        unsupervised_ratio_tmp = []

        # Initialize epoch training
        # Train per batch
        full_train_data = zip(labeled_train_data, unlabeled_train_data)
        for batch_idx, (
        (labeled_image_batch, labeled_targets), (unlabeled_image_batch, unlabeled_targets)) in enumerate(
                full_train_data):

            # Update of CTA
            if (batch_idx % 15) == 0:
                cta.update_CTA(model, labeled_train_cta_data, device)

            # Current learning rate to compute the loss combination
            lambda_unsupervised = 1

            # Set gradients to zero before start training
            model.zero_grad()

            # Train model, update weights per epoch based on the combination of labeled and unlabeled losses
            model.train()
            semi_supervised_loss, supervised_loss, unsupervised_loss, unsupervised_ratio = train_fixmatch(model,
                                                                                                          device,
                                                                                                          labeled_image_batch,
                                                                                                          labeled_targets,
                                                                                                          unlabeled_image_batch,
                                                                                                          unlabeled_batch_size,
                                                                                                          lambda_unsupervised,
                                                                                                          pseudo_label_threshold
                                                                                                          )

            # Update the weights
            semi_supervised_loss.backward()

            # Update optimizer (SGD)
            optimizer.step()

            # Stack learning process
            semi_supervised_loss_list_tmp.append(semi_supervised_loss.item())
            supervised_loss_list_tmp.append(supervised_loss.item())
            unsupervised_loss_list_tmp.append(unsupervised_loss.item())
            unsupervised_ratio_tmp.append(unsupervised_ratio)

            # Update learning rate
            scheduler.step()

            # Create EMA after warmpup
            if epoch == 10:
                ema = EMA(ema_decay, device)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema.register(name, param.data)
            elif epoch > 10:
                # Update EMA parameters
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema(name, param.data)

        # Test and compute the accuracy for the current model and exponential moving average
        model.zero_grad()
        semi_supervised_loss_list.append(np.mean(semi_supervised_loss_list_tmp))
        supervised_loss_list.append(np.mean(supervised_loss_list_tmp))
        unsupervised_loss_list.append(np.mean(unsupervised_loss_list_tmp))
        unsupervised_ratio_list.append(np.mean(unsupervised_ratio_tmp))
        print('Unsupervised Loss', unsupervised_loss_list[-1])
        print('Unsupervised ratio', unsupervised_ratio_list[-1])

        acc_ema_tmp = test_fixmatch(model, test_loader, device)
        acc_ema.append([i.item() for i in acc_ema_tmp.tolist()])
        print('Accuracy of ema', acc_ema[-1])
        # Save best model
        if cts.DATASET[0] == "SVHN":
            acc_comparison = np.mean(acc_ema[-1])
        else:
            acc_comparison = acc_ema[-1]
        if acc_comparison > best_acc:
            best_acc = acc_comparison
            final_model = model
            if not os.path.exists("best_model"):
                os.mkdir("best_model")
            string = './best_model/final_model_.pt'
            f = open("best_model_description.txt", "w+")
            f.write("Best model corresponds to epoch: " + str(epoch) + '\n')
            if cts.DATASET[0] == "SVHN":
                for c in range(cts.DATASET[4]):
                    f.write("Accuracy class " + str(c) + " is: " + str(np.array(acc_ema)[-1, c]) + '\n')
            else:
                f.write("Accuracy is: " + str(best_acc) + '\n')
            f.write("The parameters were:\n")
            f.write("n_labeled_data = " + str(n_labeled_data) + '\n')
            f.write("B = " + str(B) + '\n')
            f.write("mu = " + str(mu) + '\n')
            torch.save(final_model, string)

        if epoch % 10 == 0 and epoch != 0 and cts.DATASET[0] != "SVHN":
            epoch_range = range(epoch + 1)
            # Plot Accuracy
            plot_performance('Performance', 'Epochs', 'Accuracy', epoch_range, acc_ema, CB91_Blue)
            string_name = "Accuracy" + str(n_labeled_data) + ".png"
            plt.savefig(string_name)
            plt.close()

            # Plot Losses
            plot_performance('Semi Supervised Loss', 'Epochs', 'Loss', epoch_range, semi_supervised_loss_list,
                             CB91_Blue)
            plot_performance('Supervised Loss', 'Epochs', 'Loss', epoch_range, supervised_loss_list, CB91_Green)
            plot_performance('Unsupervised Loss', 'Epochs', 'Loss', epoch_range, unsupervised_loss_list, CB91_Red)
            string_name = "Loss" + str(n_labeled_data) + ".png"
            plt.savefig(string_name)
            plt.close()
        elif epoch % 10 == 0 and epoch != 0 and cts.DATASET[0] == "SVHN":
            epoch_range = range(epoch + 1)
            # Plot Accuracy
            for c in range(cts.DATASET[4]):
                st = "Class " + str(c)
                plot_performance(st, 'Epochs', 'Accuracy', epoch_range, np.array(acc_ema)[:, c])
            string_name = "Accuracy" + str(n_labeled_data) + ".png"
            plt.savefig(string_name)
            plt.close()

            # Plot Losses
            plot_performance('Semi Supervised Loss', 'Epochs', 'Loss', epoch_range, semi_supervised_loss_list,
                             CB91_Blue)
            plot_performance('Supervised Loss', 'Epochs', 'Loss', epoch_range, supervised_loss_list, CB91_Green)
            plot_performance('Unsupervised Loss', 'Epochs', 'Loss', epoch_range, unsupervised_loss_list, CB91_Red)
            string_name = "Loss" + str(n_labeled_data) + ".png"
            plt.savefig(string_name)
            plt.close()

    epoch_range = range(total_training_epochs)

    # Plot Accuracy
    plot_performance('Performance', 'Epochs', 'Accuracy', epoch_range, acc_ema, CB91_Blue)
    string_name = "Accuracy" + str(n_labeled_data) + ".png"
    plt.savefig(string_name)
    plt.close()

    # Plot Losses
    plot_performance('Semi Supervised Loss', 'Epochs', 'Loss', epoch_range, semi_supervised_loss_list, CB91_Blue)
    plot_performance('Supervised Loss', 'Epochs', 'Loss', epoch_range, supervised_loss_list, CB91_Green)
    plot_performance('Unsupervised Loss', 'Epochs', 'Loss', epoch_range, unsupervised_loss_list, CB91_Red)
    string_name = "Loss" + str(n_labeled_data) + ".png"
    plt.savefig(string_name)
    plt.close()

    # Print final performance with EMA
    acc_ema_final = test_fixmatch(ema, test_loader, device)
    print("Final EMA Performance: ", acc_ema_final)


if __name__ == "__main__":
    main()
