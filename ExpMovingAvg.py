# Obtained from: https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/3


class EMA:
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
