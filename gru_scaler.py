import torch

class BaseScaler:
    """Abstract scaler class."""

    def fit(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError

class StandardScaler(BaseScaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        # data: torch.tensor [T, num_features]
        self.mean = data.mean(dim=0)
        self.std = data.std(dim=0)
        return self

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

class MinMaxScaler(BaseScaler):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        self.min = data.min(dim=0).values
        self.max = data.max(dim=0).values
        return self

    def transform(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform(self, data):
        return data * (self.max - self.min) + self.min
