import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

#Ensure all nltk data is correclty downloaded

try :
    stopwords.words('english') #stop words are just common words that are not useful for analysis
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet') #wordnet is a database of english words and their meanings 
except LookupError:
    nltk.download('wordnet')

    #wordnet is used to lemmatize words, meaning just processing their root form



    #Data Paths:
    DATA_PATH_ALLSIDES = 'Data/AllSidesMedia/allsides_balanced_news_headlines-texts.csv' #csv
    DATA_PATH_BABE_SG1 = 'Data/BABE/data/final_labels_SG1.csv'# csv
    DATA_PATH_BABE_SG2 = 'Data/BABE/data/final_labels_SG2.csv' #csv
    DATA_PATH_MBIC = 'Data/MBIC/labeled_dataset.xlsx' #excel
    OUTPUT_PATH = 'processed/combined_data.csv' #csv



    def clean_text(text):
        """
        Cleans texyt by lowercasing, removing puncation, and removing stop words.
        Not sure on including lemmatization, yet.

        arg: text: str - the text to clean
        returns: str- cleaned text
        """
        if not isinstance(text,str):
            return ""
        #turn to lowercase
        text = text.lower()

        #remove punctuation
        text = re.sub(r'[^\w\s]', '', text) 
        # the ^ means not, \w means word character, \s means space 

        #remove stop words

        stop_words = set(stopwords.words('english'))
        text = ' '.join([word for word in text.split() if word not in stop_words]) #creates a new string with only non stop words
        return text 

def load_allsides(path):
    """
    loads the AllSides dataset, standardizes labels, and selects relevant columns.
    """
    print(f"Loading AllSides data from {path}...")
    try:
        df = pd.read_csv(path)
        # standardize the label column
        df['label'] = df['bias_rating'].str.lower()
        # select and rename columns to the unified format
        df = df[['source', 'text', 'label']]
        # filter for the labels we care about
        df = df[df['label'].isin(['left', 'center', 'right'])]
        print(f"Loaded {len(df)} articles from AllSides.")
        return df.dropna(subset=['text', 'label'])
    except  FileNotFoundError:
        print(f"Warning: AllSides file not found at {path}. Skipping.")
        return pd.DataFrame(columns=['source', 'text', 'label'])



def load_babe(path_sg1, path_sg2):
    """
    Loads and combines the BABE datasets, standardizes labels, and selects relevant columns.
    """
    print(f"Loading BABE data from {path_sg1} and {path_sg2}...")
    try:
        df1 = pd.read_csv(path_sg1, delimiter=';')
        df2 = pd.read_csv(path_sg2, delimiter=';')
        df = pd.concat([df1, df2], ignore_index=True)
        # standardize the label column
        df['label'] = df['type'].str.lower()
        # select and rename columns
        df = df[['outlet', 'text', 'label']].rename(columns={'outlet': 'source'})
         # filter for the labels we care about
        df = df[df['label'].isin(['left', 'center', 'right'])]
        print(f"Loaded {len(df)} articles from BABE.")
        return df.dropna(subset=['text', 'label'])
    except FileNotFoundError:
        print(f"Warning: BABE file(s) not found. Skipping.")
        return pd.DataFrame(columns=['source', 'text', 'label'])


def load_mbic(path):
    """
    Loads the MBIC dataset, standardizes labels, and selects relevant columns.
    Uses the correct MBIC columns: 'outlet', 'sentence', and 'type'.
    """
    print(f"Loading MBIC data from {path}...")
    try:
        df = pd.read_excel(path)
        df['label'] = df['type'].str.lower()
        df = df[['outlet', 'sentence', 'label']].rename(columns={'outlet': 'source', 'sentence': 'text'})
        df = df[df['label'].isin(['left', 'center', 'right'])]
        print(f"Loaded {len(df)} articles from MBIC.")
        return df.dropna(subset=['text', 'label'])
    except FileNotFoundError:
        print(f"Warning: MBIC file not found at {path}. Skipping.")
        return pd.DataFrame(columns=['source', 'text', 'label'])
    except KeyError:
        print(f"Warning: MBIC file at {path} does not have the expected columns ('outlet', 'sentence', 'type'). Skipping.")
        return pd.DataFrame(columns=['source', 'text', 'label'])


def load_all_data():
    # loads all datasets
    print("Data loading started:")

    # load datasets
    allsides_df = load_allsides(DATA_PATH_ALLSIDES)
    babe_df     = load_babe(DATA_PATH_BABE_SG1, DATA_PATH_BABE_SG2)
    mbic_df     = load_mbic(DATA_PATH_MBIC)

    # combine datasets
    combined_df = pd.concat([allsides_df, babe_df, mbic_df], ignore_index=True)

    if not combined_df.empty:
        print("Cleaning data:")
        combined_df['text'] = combined_df['text'].apply(clean_text)

        # remove duplicates
        combined_df.drop_duplicates(subset=['text'], inplace=True)

        # ensure output directory exists
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

        # save processed data
        combined_df.to_csv(OUTPUT_PATH, index=False)
        print(f"Processed data saved to {OUTPUT_PATH}")
        print(f"Total articles: {len(combined_df)}")
    else:
        print("No data loaded. Please check data paths and file formats.")

if __name__ == '__main__':
    load_all_data()



