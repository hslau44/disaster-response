# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv sqlite:///data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py {host}{port}`

    - If `host` or `port` is not defined, the apps will direct you to `localhost` and `8899` respectively.
    - For default setting, as the app finish setting up, go to http://localhost:8899/
