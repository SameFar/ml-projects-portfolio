from src import *

def main():
    if input('Enter "t" to train or anything else to predict tomorrows nvidea stock: ') == 't':
        train_data, test_data = train_test_split()
        
        train_loader = make_loader(train_data, timestep=50)
        eval_loader  = make_loader(test_data, timestep=50)
        
        model = StockLSTM(input_size=3, hidden_size=50, num_layers=1)

        train(train_loader, eval_loader, model)
    
    else:
        print(predict_tomorrow_open())

if __name__ == '__main__':
    main()
    