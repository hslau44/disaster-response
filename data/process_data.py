import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def str_to_dic(string):
    """
    transform the column from string to dictionary for processing

    Input:
    string: str; original representation of categories

    Return:
    dic: dic; hasable representation of categories

    Example:
    string = 'related-1;request-0;offer-0;aid_related-0'
    dic = str_to_dic(string)
    >> dic
    {'related': 1,
     'request': 0,
     'offer': 0,
     'aid_related': 0,}
    """
    dic = {}
    for i in string.split(';'):
        category,code = i.split('-')
        dic[category] = int(code)
    return dic

def load_data(messages_filepath, categories_filepath):
    """
    Return pandas.dataframe files of message and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(messages,categories):
    """
    Transfrom categories into dummy variables, and merge with message to form a new dataframe
    """
    categories['categories'] = categories['categories'].apply(str_to_dic)
    # select the first row to obtain all category names
    row_num = 0
    row = categories['categories'][row_num]
    category_colnames = list(row.keys())
    # create empty table that has columns: category names, and merge with categroies
    categories = pd.concat([categories,
                            pd.DataFrame(columns=category_colnames)
                           ],axis=1)
    # for each category
    for col in category_colnames:
        # filling the column X by extracting the value of the hash table in 'categories' with key X
        categories[col] = categories['categories'].apply(lambda x: x[col])

    # removing rows with any unbinarized value
    categories = categories[(categories[category_colnames].isin([0,1])).all(axis=1)]
    # drop 'categories' in categories
    categories = categories.drop(['categories'],axis=1)
    df = pd.merge(messages,categories,on='id',how='inner')
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Save data with table name  'DisasterResponse'
    """
    engine = create_engine(database_filename) # 'sqlite:///' +
    df.to_sql('DisasterResponse', engine, index=False)
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
