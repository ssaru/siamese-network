from torchvision import transforms
from base import BaseDataLoader
from data_loader.datasets import SiameseNetworkDataset
import torchvision.datasets as torch_dataset

class SiameseNetworkDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor()
            ])
        self.data_dir = data_dir
        self.folder_dataset = torch_dataset.ImageFolder(root= self.data_dir)
        self.siamese_dataset = SiameseNetworkDataset(self.folder_dataset, transform=trsfm, should_invert=False)
        super(SiameseNetworkDataLoader, self).__init__(self.siamese_dataset, batch_size, shuffle, validation_split, num_workers)
