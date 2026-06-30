from src import *
import logging

def main():
    make_logger()
    while True:
        c = input('Enter "t" to train or "p" to predict specified nvidea stock, neither to exit: ').lower()
        if c == 't':

            logging.debug('Making data loaders')
            train_data, test_data = train_test_split()
            train_loader = make_loader(train_data, timestep=50)
            eval_loader  = make_loader(test_data, timestep=50)
            
            logging.debug('Starting Training')
            model = StockLSTM(input_size=3, hidden_size=50, num_layers=1)
            train(train_loader, eval_loader, model)
        
        elif c =='p':
            logging.info(f"Model predicted: {predict_tomorrow_open()}")

        else:
            break

if __name__ == '__main__':
    main()
    