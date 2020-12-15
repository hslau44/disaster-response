import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def str_to_dic(string):
    # transform the column from string to dictionary for processing
    dic = {}
    for i in string.split(';'):
        category,code = i.split('-')
        dic[category] = int(code)
    return dic

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(messages,categories):
    categories['categories'] = categories['categories'].apply(str_to_dic)
    row_num = 0 # select the first row
    row = categories['categories'][row_num]
    category_colnames = list(row.keys())
    categories = pd.concat([categories,
                            pd.DataFrame(columns=category_colnames)
                           ],axis=1)
    for col in category_colnames:
        categories[col] = categories['categories'].apply(lambda x: x[col])

    categories = categories[(categories[category_colnames].isin([0,1])).all(axis=1)]
    categories = categories.drop(['categories'],axis=1)
    df = pd.merge(messages,categories,on='id',how='inner')
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
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
