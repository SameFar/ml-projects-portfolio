import logging
import torch
from src import (
    IntelCNN,
    get_test_loader,
    get_train_val_loaders,
    load_model,
    make_logger,
    save_model,
    test_model,
    train_model,
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

make_logger()
logging.debug("Imports done")


def main():
    while True:
        # TRAINING

        c1 = input("Train model? y/n: ").lower()
        if c1 == "y":
            model = IntelCNN().to(device)
            logging.debug("Fetching Data...")
            train_loader, val_loader = get_train_val_loaders()
            logging.debug("Data Loaded, Starting Training...")

            m = train_model(model, train_loader, val_loader, device)

            c2 = input("Save trained model? y/n: ").lower()

            if c2 == "y":
                save_model(m)

            elif c1 != "n":
                logging.error("Invalid input, Stopping Program")
                break

        elif c1 != "n":
            logging.error("Invalid input")
            break

        # TESTING

        c3 = input("Test model? y/n: ").lower()
        if c3 == "y":
            logging.debug("Loading Data...")
            test_loader = get_test_loader()
            logging.debug("Data Loaded, Starting Testing...")

            model = test_model(load_model(), test_loader, device)

            logging.debug("Testing done!")

        elif c3 != "n":
            logging.error("Invalid input")
            break

        break
    logging.debug("Ending Program")


if __name__ == "__main__":
    main()
