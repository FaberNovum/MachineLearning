import torch
import matplotlib.pyplot as plt

# create a tensor
x = torch.linspace(-10, 10, 100)

# use logistic activation function on tensor
y = torch.tanh(x)

# plot results
plt.plot(x.numpy(), y.numpy(), color="red")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Tanh Activation Function")
plt.show()
