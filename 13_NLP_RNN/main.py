import logging
from pathlib import Path

from src import (
    CharRNN,
    NamesDataset,
    allowed_chars,
    guess_name_origin,
    make_logger,
    train,
    visualise,
)


def main():
    while True:
        make_logger()
        model_path = Path(__file__).resolve().parent / "model" / "model.pt"
        c = input('Press "y" to train or "n" to guess, anything else to exit ')

        if c == "y":
            alldata = NamesDataset()
            logging.debug("Data loaded")
            rnn = CharRNN(len(allowed_chars), 128, len(alldata.labels_uniq))
            loss = train(rnn, alldata, export_path=model_path)
            visualise(loss)
            return

        elif c == "n":
            print(guess_name_origin(input("Enter name to guess: "), model_path))

        else:
            logging.critical("Stopping program")
            break


if __name__ == "__main__":
    main()
