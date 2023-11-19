from torchvision import datasets, transforms
import torch


class MNIST_dataset():
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

    def get_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(self.path, download=True, train=True, transform=transform)
        testset = datasets.MNIST(self.path, download=True, train=False, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=True)
        return trainloader, testloader
    

if __name__ == '__main__':
    dataset = MNIST_dataset('./data', 2)
    trainloader, testloader = dataset.get_dataset()
    # print one image and its label
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    print(images.shape)
    print(labels.shape)
    print(images[0].shape)
    print(labels[0].shape)
    print(labels[0])
    