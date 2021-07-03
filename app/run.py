#Import libraries 
import pandas as pd
import nltk, re, plotly, json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar,Pie
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib



app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Creates a new feature for the ML model by extracting the starting verb of a sentence. 
    
    Parameters:
    class: Base class for all estimators in scikit-learn.
    class: Mixin class for all transformers in scikit-learn.

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
    
  
def tokenize(text):
    """
    Tokenize and preprocess the text of the messages (English version) for ML modeling. 
    
    Parameters:
    list: List of all the text messages in the database (English version).
    Returns:
    list: List of the preprocess and clean tokenized text, ready for ML modeling. 
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

# Load the  DisasterResponse dataset from the SQL database 
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# Load the model we have trained 
model = joblib.load("../models/classifier.pkl")


# Index webpage which displays cool visualizations 
# and receives the input text provided by users for the model prediction 
@app.route('/')
@app.route('/index')
def index():
    
    # Create the datasets needed for visualizations 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    categories_name = df.iloc[:,4:].columns
    categories_counts = (df.iloc[:,4:] != 0).sum().values

    line_categories_counts = df[df.columns[4:]].sum().sort_values(ascending=False).values
    line_categories_names = df[df.columns[4:]].sum().sort_values(ascending=False).index
    
    # Create cool visualizations for the webpage
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=categories_name,
                    y=categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories (Barplot)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        },
        {
            'data': [
                Pie(
                    labels=line_categories_names,
                    values=line_categories_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories (Pie)',
                'yaxis': {
                    'title': "Categories"
                },
                'xaxis': {
                    'title': "Count"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()