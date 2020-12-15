# Disaster Response Pipeline Project

This repository is Udacity Data engineering project. To help responding the messages received during the disaster, this web app is built to help the user(Analyst) to classify a piece message into specific category(ies), so that the user can forward the message to the corresponding disaster relief agencies.   

### files

1. app
    - template: folder storing front end template (html) of the app
    - run.py: python script to run the apps

2. data
    - disaster_messages.csv : corresponding categories of the message
    - disaster_categories.csv : original form of the message
    - (optional) DisasterResponse.db : SQLite database storing ETL data, create if not exist.
    - process_data.py : python script of the ETL pipeline; ETL loads the original data, merge and clean the data and save the data into SQLite database

3. model
    - classifier.pkl : joblib file of the classifier (default: RandomForestClassifier)
    - pipeline.pkl : joblib file of the data processing pipeline
    - train_classifier.py: python script for creating and training pipeline and classifier; it loads the SQLite database, splits the dataset into training and test set, fits the pipeline and classifier with the training set, evaluates the classifier with the test set, and save both of training pipeline and classifier


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models True`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py {host}{port}`

    - If `host` or `port` are not defined, the apps will direct to `localhost` and `8899` respectively.
    - For default setting, as the app finish setting up, go to http://localhost:8899/

3. Type the message in the search engine.

    - User can defined the genre of the message by adding `|` between the message and genre:
      `{message}|{genre}`
    - the available genre are `direct` (default), `social`, and `news`.
