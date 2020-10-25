# Function obtained from: https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3

class EMA():
    def __init__(self, mu, device):
        self.mu = mu
        self.shadow = {}
        self.device = device

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, name, x):
        assert name in self.shadow
        new_average = (1.0 - self.mu) * x + self.mu * self.shadow[name].to(self.device)
        self.shadow[name] = new_average.clone()
        return new_average

    def copy_to(self, parameters):
        """
        Copies current parameters into given collection of parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored moving averages.
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)


