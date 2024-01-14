from dataset.get_data import read_dataset, get_train_data
from train.train_utils import train_model
from eval.eval_utils import test_model
from numpy import save, load


if __name__ == '__main__':

    print("Loading and preparing data...\n")
    X_train, X_test, y_train, y_test = read_dataset()
    X_train, X_valid, y_train, y_valid = get_train_data(X_train, y_train)
    #save('./X_test.npy', X_test)
    #save('./y_test.npy', y_test)
    print("Data loaded! :)\n\n")

    print("Training model...\n")
    class_weights = train_model(X_train, X_valid, y_train, y_valid, model_typ='custom', weighted=False)
    print('Model trained! :)\n\n')

    X_test = load('./X_test.npy')
    y_test = load('./y_test.npy')
    print("Test model...\n")
    test_model(X_test, y_test, model_typ='custom')

