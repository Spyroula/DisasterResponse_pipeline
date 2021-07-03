"""
Train Classifier script for Disaster Resoponse Project
Udacity - Data Science Nanodegree
To run the ML pipeline that trains classifier and saves the model try the following command: 
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Arguments:
    1) The SQLite database path. It should contain the pre-processed data. 
    2) The name of the pickle file you use to save the ML model. 
"""

# Ιmport libraries
import sys
import pandas as pd
import numpy as np
import pickle
from sqlalchemy import create_engine
import re
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV
from scipy.stats import hmean
from scipy.stats.mstats import gmean

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    """
    Loads data from an sql database and returns feature and target variables X and Y and categories names.
    
    
    Parameters:
    database_filepath (str): The path to the SQLite database.
    
    
    Returns:
    X (DataFrame): Returning the X (feature) dataFrame.
    Y (DataFrame): Returning the Y (feature) dataFrame.
    categories_names (list): the list of the category names (used for data visualization in the app).
   """
    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse',engine)
    X = df['message']
    Y = df.iloc[:,4:]
    categories_names = Y.columns
    return X, Y, categories_names

def tokenize(text):
    """
    Tokenize and preprocess the text of the messages (English version) for ML modeling. 
    
    Parameters:
    text (list): List of all the text messages in the database (English version).
    Returns:
    final_tokens (list): List of the preprocess and clean tokenized text, ready for ML modeling. 
    """
    regex_url = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls_detected = re.findall(regex_url, text)
    for url in urls_detected:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    final_tokens = []
    for token in tokens:
        final_token = lemmatizer.lemmatize(token).lower().strip()
        final_tokens.append(final_token)

    return final_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Creates a new feature for the ML model by extracting the starting verb of a sentence. 
    
    Parameters:
    BaseEstimator (class): Base class for all estimators in scikit-learn.
    TransformerMixin (class): Mixin class for all transformers in scikit-learn.
    """

    def starting_verb(self, text):
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            tagged_tokens = nltk.pos_tag(tokenize(sentence))
            token, tag = tagged_tokens[0]
            if tag in ['VB', 'VBP'] or token == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build a ML model through an appropriate NLP pipeline
    
    Returs:
    mdl (obj): ML Pipeline that process text messages with the best well-known NLP practice and apply a classifier.
    """
    parameters = {
    'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    'features__text_pipeline__vect__max_df': (0.75, 1.0),
    'features__text_pipeline__vect__max_features': (None, 5000),
    'features__text_pipeline__tfidf__use_idf': (True, False),
    }


    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    mdl = GridSearchCV(pipeline, param_grid=parameters,verbose = 2, n_jobs = -1)
    
    return mdl

def multioutput_f_score(y_true,y_pred,beta=1):
    """
    This is a custom performance metric representing a geometric mean 
    of the fbeta_score, calculated on each label.
    
    It supports multi-label and multi-class problems and it is capable to  exclude
    trivial solutions and on purpose under-estimate a average of fbeta_scores.
    The purpose of using this custom performance metric is to avoid issues occured when  
    dealing with multi-class/multi-label imbalanced datasets.
    
        
    Parameters:
    y_true (DataFrame): The true labels dataframe y_true.
    y_pred (DataFrame): The predictions dataframe y_prod. 
    beta (int): The beta value of the fscore metric.
    
    Returns:
    f1_score (array): The customised f1_score. 
    """
    scores = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        scores.append(score)
    f1_score_numpy = np.asarray(scores)
    f1_score_numpy = f1_score_numpy[f1_score_numpy<1]
    f1_score = gmean(f1_score_numpy)
    return  f1_score

def evaluate_model(model, X_test, Y_test, categories_names):
    """
    Apply our ML pipeline to a test set and returns the model performance
    in terms of accuracy and f1_score o

    Parameters:
    model: ML Pipeline that process text messages with the best well-known NLP practice and apply a classifier.
    X_test (DataFrame): The test features, X_test. 
    Y_test (DataFrame): The test labels Y_test.
    categories_names (list): the list of the category names (used for data visualization in the app). 
    """
    Y_pred = model.predict(X_test)
    
    multioutput_f1 = multioutput_f_score(Y_test,Y_pred, beta = 1)
    avg_accuracy = (Y_pred == Y_test).mean().mean()

    print('Average total accuracy {0:.2f}% \n'.format(avg_accuracy*100))
    print('F1_score (custom performance metric) {0:.2f}%\n'.format(multioutput_f1*100))

    Y_pred_pd = pd.DataFrame(Y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('------------------------------------------------------\n')
        print('Model performance for the category: {}\n'.format(column))
        print(classification_report(Y_test[column],Y_pred_pd[column]))
    
    pass


def save_model(model, model_filepath):
    """
    Saving the ML trained model as Pickle file, to be loaded later.
    
    Parameters:
    model (obj): The ML trained model, a scikit-learn Pipeline object
    model_filepath (str): The destination path to save the .pkl file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    """
    The Main function applies the Machine Learning Pipeline:
        1) Extract data from SQLite database
        2) Train a ML model on the training set
        3) Evaluate the trained model on the test set and estimates its performance 
        4) Save  the trained ML model as Pickle file 
    
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, categories_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()