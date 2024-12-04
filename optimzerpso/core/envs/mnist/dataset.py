from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class MNISTDataset:

    def __init__(self, batch_size: int = 64):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.train_dataset = datasets.MNIST(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
        
        self.test_dataset  = datasets.MNIST(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        
        self.test_loader  = DataLoader(self.test_dataset,
                                      batch_size=batch_size,
                                      shuffle=False)
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_test_loader(self):
        return self.test_loader
