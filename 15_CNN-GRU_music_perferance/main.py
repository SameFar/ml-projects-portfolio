from src import *
import logging
from pathlib import Path

def main():
    make_logger()
    songs_dir = Path(__file__).resolve().parent.parent / 'songs'
    pred_dir = songs_dir / 'pred'
    model = PannGru()

    while True:

        c = input("""

    1. Get song verdit from video url
    2. Scrape youtube playlist for favoured songs
    3. Train model
    4. Exit

    (For first time use, first run 2 and then 3)
                
    Choice: """)
        
        if c.isdigit():
            c = int(c)
        else:
            logging.error("Invalid input, restarting program")
            continue

        match c:
            case 1:
                clear_directory(pred_dir)
                url = input('Youtube URL: ')
                audio_scraper(url, pred_dir)

                dataloader = make_dataloader(pred_dir)
                model = load_model(model)
                
                print(predict_preferences(model, dataloader))
            case 2:
                audio_scraper("https://www.youtube.com/playlist?list=PLPxXfwPeO7ZY", songs_dir)
            case 3:
                dataloader = make_dataloader(songs_dir)
                train(model, dataloader)
            case 4:
                break
            case _:
                logging.error("Invalid input, restarting program")
                
                continue




if __name__ == '__main__':
    main()