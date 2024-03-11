import dvc.api

from helper import load_data
from sklearn.metrics import accuracy_score
from dvclive import Live
from mlem.api import load
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def load_model(path: str):
    """Load model from path"""
    return load(path)


def evaluate() -> None:
    """Evaluate model and log metrics"""
    params = dvc.api.params_show()
    with Live(save_dvc_exp=True, resume=True) as live:
        print('Loading data...')
        X_test = load_data(f"{params['data']['preprocessed']}/test.npy")
        y_test = load_data(f"{params['data']['preprocessed']}/test_labels.npy")
        model = load_model(params["model"])
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        #print(f"The model's accuracy is {accuracy}")
        precision,recall,fscore,support=precision_recall_fscore_support(y_test,y_pred,average='macro')
        print('\nClassification Metrics:')
        print('Precision : {}'.format(precision))
        print( 'Recall    : {}'.format(recall))
        print('F-score   : {}'.format(fscore))
        print('Accuracy  : {}'.format(accuracy))
        print('======================================================')
        live.log_metric("accuracy", accuracy)

if __name__ == "__main__":
    evaluate()