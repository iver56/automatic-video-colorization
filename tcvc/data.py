from torch.utils.data import DataLoader

from tcvc.dataset import DatasetFromFolder


def get_dataset(root_dir, use_line_art=True):
    return DatasetFromFolder(root_dir, use_line_art)


def create_iterator(sample_size, sample_dataset):
    while True:
        sample_loader = DataLoader(
            dataset=sample_dataset, batch_size=sample_size, drop_last=True
        )

        for item in sample_loader:
            yield item
