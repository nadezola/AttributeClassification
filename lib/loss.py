import torch

class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.crit1 = torch.nn.CrossEntropyLoss()
        self.crit2 = torch.nn.CrossEntropyLoss()
        self.crit3 = torch.nn.CrossEntropyLoss()
        self.crit4 = torch.nn.CrossEntropyLoss()

    def forward(self, outputs, batch):
        self.loss1 = self.crit1(outputs[0], batch[:, :5])
        self.loss2 = self.crit2(outputs[1], batch[:, 5:7])
        self.loss3 = self.crit3(outputs[2], batch[:, 7:10])
        self.loss4 = self.crit4(outputs[3], batch[:, 10:14])
        return self.loss1, self.loss2, self.loss3, self.loss4

    def backward(self):
            self.loss1.backward()
            self.loss2.backward()
            self.loss3.backward()
            self.loss4.backward()

