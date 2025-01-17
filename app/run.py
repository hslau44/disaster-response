import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
from models.train_classifier import Column_extractor,transfrom_text_query,tokenize


def category_info(df):
    """
    Return the amount of data in each category(normalized), and the category names.

    Input:
    df: pandas.DataFrame

    Return:
    category_counts: pandas.core.series.Series; amount of data in each category(normalized)
    category_names: list; the category names
    """
    y = df.iloc[:,4:].values
    category_names = df.columns[4:].tolist()
    dic = {}
    total = y.shape[0]
    for i in range(y.shape[1]):
        dic[category_names[i]] = y[y[:,i] == 1].shape[0]/total
    category_counts = pd.Series(dic)
    return category_counts,category_names


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()
#
#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)
#
#     return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# con = sqlite3.connect('./data/DisasterResponse.db')
# df = pd.read_sql("SELECT * FROM DisasterResponse", con)

# load model
pipeline = joblib.load("../models/pipeline.pkl")
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts,category_names = category_info(df)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Category',
                'yaxis': {
                    'title': "Proportion"
                },
                'xaxis': {
                    'title': "Category"
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
    inputs = transfrom_text_query(query)
    classification_labels = model.predict(pipeline.transform(inputs))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    if len(sys.argv) == 3:
        host,port = sys.argv[1:]
    else:
        host,port = 'localhost',8899

    app.run(host=host, port=port, debug=True)


if __name__ == '__main__':
    main()
