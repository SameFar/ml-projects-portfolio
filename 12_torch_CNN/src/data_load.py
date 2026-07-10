import kagglehub
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from pathlib import Path

path = Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))

transformer = transforms.Compose(
    [
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def get_train_val_loaders(val_split=0.2, batch_size=32):
    """
    Loads the training directory and splits it into
    Training and Validation DataLoaders.

    Args:
        val_split:  How much of the data is split into validation (default = 0.2),
        batch_size: How much to load at one time (default=32)

    Returns:
        train_loader: Train Dataloader,
        val_loader:   Validation Dataloader
    """
    train_dataset = datasets.ImageFolder(
        root=path / "seg_train" / "seg_train", transform=transformer
    )

    total_images = len(train_dataset)
    val_size = int(total_images * val_split)
    train_size = total_images - val_size

    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    # Create loaders (Shuffle train data, keep val data static)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(batch_size=32):
    """
    Loads the testing directory and returns the Test DataLoader.

    Args:
        batch_size: How much to load at one time (default=32)
    """
    # Load the raw testing folder
    test_dataset = datasets.ImageFolder(
        root=path / "seg_test" / "seg_test", transform=transformer
    )

    # Create test loader (No need to shuffle final test data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader
