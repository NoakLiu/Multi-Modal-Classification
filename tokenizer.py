import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as sw

class Tokenizer:
    def __init__(self):
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        self.stop_words = sw.words('english')
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_and_preprocess(self, df_train, df_test):
        df_train['Caption'] = df_train['Caption'].apply(lambda x: self.transform_text(x, False))
        df_test['Caption'] = df_test['Caption'].apply(lambda x: self.transform_text(x, False))

        x_train = df_train[['ImageID', 'Caption']]
        y_train = df_train['Labels'].apply(lambda x: x.split()).apply(lambda x: [int(a) for a in x])
        x_test = df_test[['ImageID', 'Caption']]

        # Additional preprocessing if necessary

        return x_train, y_train, x_test, None

    def transform_text(self, text, extract_nouns=False):
        is_noun = lambda pos: pos[:2] == 'NN'
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        text = re.sub(r'[0-9]+', '', text)
        text = word_tokenize(text)
        if extract_nouns:
            text = [word for (word, pos) in nltk.pos_tag(text) if is_noun(pos)]
        text = [self.lemmatizer.lemmatize(a) for a in text if a not in self.stop_words]
        return text
