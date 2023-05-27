import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# load MNIST
transform = transforms.ToTensor()
train_set = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_set = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
