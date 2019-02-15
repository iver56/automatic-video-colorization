
from os.path import join
from dataset import DatasetFromFolder
from torch.utils.data import DataLoader

def get_training_set(root_dir):
    train_dir = join(root_dir, "Train")

    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "Test")

    return DatasetFromFolder(test_dir)

def get_val_set(root_dir):
    val_dir = join(root_dir, "Val")

    return DatasetFromFolder(val_dir)

def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = DataLoader(
            dataset= sample_dataset,
            batch_size=sample_size,
            drop_last=True
        )

        for item in sample_loader:
            yield item
