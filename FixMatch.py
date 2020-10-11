from WideResNet import *

if __name__ == "main":
    d = 28  # Network depth
    k = 2  # Network width factor
    strides = [1, 1, 2, 2]
    net = WideResNet(d=d, k=k, n_classes=10, input_features=3, output_features=16, strides=strides)

    # verify that an output is produced
    sample_input = torch.ones(size=(1, 3, 32, 32), requires_grad=False)
    net(sample_input)

    # Summarize model
    summary(net, input_size=(3, 32, 32))