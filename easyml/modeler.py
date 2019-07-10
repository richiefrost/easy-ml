from sklearn.metrics import recall_score, precision_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pickle

class Modeler:
    """ 
    Modeler makes repetitive classification tasks simpler in Scikit-Learn.
    Out of the box, it gives you common metrics such as recall, precision, accuracy, and AUC ROC.
  
    Example:
    --------------------------
    Let's say you want to run a classification experiment. Here's how you do it:
    
    >>> from easyml import Model
    >>> modeler = Modeler(df, features, label, model, verbose=True)
    >>> modeler.fit()
    
        Getting metrics for train:
        -----------------------------------------
        recall_score: 1.0
        precision_score: 1.0
        roc_auc_score: 1.0
        accuracy_score: 1.0

        Getting metrics for test:
        -----------------------------------------
        recall_score: 0.9166666666666666
        precision_score: 0.9166666666666666
        roc_auc_score: 0.8869047619047619
        accuracy_score: 0.8947368421052632

        Getting metrics for full:
        -----------------------------------------
        recall_score: 0.9831932773109243
        precision_score: 0.9831932773109243
        roc_auc_score: 0.9774456952592357
        accuracy_score: 0.9789103690685413

    """

    def __init__(self, df, features, label, model, verbose=False):
        '''
        df = Pandas dataframe
        features = Feature names array
        label = label column in dataframe
        model = initialized model for training
        '''
        self._df = df
        self._features = features
        self._label = label
        self._model = model
        self._metrics = {}
        self._verbose = verbose
        
    def fit(self):
        X, y = self._df[self._features], self._df[self._label]
        X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, y, test_size=0.2)
        self._model.fit(X_tr, Y_tr)
        
        y_pred_train = self._model.predict(X_tr)
        y_pred_test = self._model.predict(X_ts)
        y_full = self._model.predict(X)
        
        self._set_metrics(Y_tr, y_pred_train, 'train')
        self._set_metrics(Y_ts, y_pred_test, 'test')
        self._set_metrics(y, y_full, 'full')
        
    def _set_metrics(self, y_true, y_pred, data_type):
        '''
        Data type can be train, test, or full
        '''
        if self._verbose:
            print('Getting metrics for {}:'.format(data_type))
            print('-----------------------------------------')
        self._metrics[data_type] = {}
        funcs = [recall_score, precision_score, roc_auc_score, accuracy_score]
        for func in funcs:
            metric_name = func.__name__
            metric_score = func(y_true, y_pred)
            if self._verbose:
                print('{}: {}'.format(metric_name, metric_score))
            self._metrics[data_type][metric_name] = metric_score
        if self._verbose:
            print()
        
    def to_pickle(self, filename):
        if self._verbose:
            print('Saving to {}'.format(filename))
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def get_model(self):
        return self._model

    def get_metrics(self):
        return self._metrics

    def get_features(self):
        return self._features