import torch
import matplotlib.pyplot as plt

def predict(net, x, y):
    y_pred = net.forward(x)
    plt.plot(x.numpy(), y.numpy(), '--', label='Ground truth')
    plt.plot(x.numpy(), y_pred.detach().numpy(), '*', c='r', label='Prediction')
    plt.legend(loc='upper left')
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.show()

def target_function(x):
    return torch.sin(x+2) * 3**x

def metric(pred, target):
    return ((pred - target) ** 2).mean()


class RegressionNet(torch.nn.Module):
    def __init__(self, hidden_size):
        super(RegressionNet, self).__init__()
        self.layer1 = torch.nn.Linear(1, hidden_size)
        self.act1 = torch.nn.ReLU() 
        self.layer2 = torch.nn.Linear(hidden_size, int(hidden_size / 2)) 
        self.act2 = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(int(hidden_size / 2), 1)
    
    def forward(self, x):
        z = self.layer1(x)
        z = self.act1(z)
        z = self.layer2(z)
        z = self.act2(z)
        output = self.output_layer(z)
        return output

net = RegressionNet(512)  

x_train = torch.linspace(-10, 5, 100)
y_train = target_function(x_train)
noise = torch.randn(y_train.shape) / 20
y_train = y_train + noise
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(-10, 5, 100)
y_validation = target_function(x_validation)
x_validation.unsqueeze_(1)
y_validation.unsqueeze_(1)

optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)  

for epoch_index in range(21930):
    optimizer.zero_grad()
    y_pred = net.forward(x_train)
    loss_value = metric(y_pred, y_train)
    loss_value.backward()
    optimizer.step()

    with torch.no_grad():
        validation_error = metric(net.forward(x_validation), y_validation)

    if (epoch_index + 1) % 50 == 0:
        print(f'Epoch {epoch_index + 1}, Train loss: {loss_value.item():.4f}, Validation loss: {validation_error.item():.4f}')

predict(net, x_validation, y_validation)