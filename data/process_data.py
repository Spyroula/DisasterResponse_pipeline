"""
Preprocess data for Disaster Resoponse Project
Udacity - Data Science Nanodegree
To run the ETL pipeline that cleans data and stores in database try the following command: 
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db 
Arguments:
    1) The path to the .CSV file with the users messages (e.g. disaster_messages.csv)
    2) The path to the .CSV file containing the categories of the meassages (e.g. disaster_categories.csv)
    3) The path to the destination of the SQLite database (e.g. data/DisasterResponse.db)
"""
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load the messages data with their categories 
    
    Parameters:
    messages_filepath (str): The path to the .CSV file with the disaster messages 
    categories_filepath (str): The path to the .CSV file with the categories
    Returns:
    df (DataFrame): The merged dataset with the disaster messages and their categories
    """
    #Load messages and categories 
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge the messages and categories datasets
    df = messages.merge(categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Clean and preprocess the data of the merged dataset
    
    Parameters:
    df (DatFrame): The dataframe containing messages and categories
    Returns:
    df (DatFrame): The dataframe containing cleaned and preprocessed messages and categories
    """

    # Split categories into separate colums resulting in a dataframe of 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # Select the first row of the categories 
    row = categories.iloc[0,:]
    # Extract a list of the new column names for categories from the first row.
    category_colnames = row.apply(lambda x: x[:-2])
    # Rename the column names of categories
    categories.columns = category_colnames
    # Convert category values to  numbers 0 or 1.
    for column in categories:
        # Set each value as the last character of the string
        categories[column] = categories[column].str[-1]
        # Convert the column type from string to numeric
        categories[column] = categories[column].astype(int)
    categories.replace(2, 1, inplace=True)
    # Drop the original categories column from df dataframe
    df.drop('categories', axis=1, inplace = True)
    # Concatenate the original df dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)
    # Drop duplicates
    df = df.drop_duplicates()
    return df
    

def save_data(df, database_filename):
    """
    Save the clean and preprocessed data to the SQLite database 
    
    Parameters:
    df (DataFrame): Clean and preprocessed dataframe containing messages and categories 
    database_filename (str): The path to the destination of the SQLite  database
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine,if_exists = 'replace', index=False)  


def main():
    """
    Main function which will implement the ETL pipeline with the following three actions:
        1) Load messages and categories data from the .csv files. 
        2) Clean and preprocess the categories Data
        3) Save data to SQLite database
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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