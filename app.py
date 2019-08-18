# Import Flask and other Packages 
from flask import Flask, render_template, url_for, request
import pandas as pd  #  to read and manipulate .csv files
import pickle  # to save and load trained models and columns
from sklearn.feature_extraction.text import CountVectorizer # to clean text data or files
from sklearn.naive_bayes import MultinomialNB # our classifier algotithm
# Other Algorithm includes
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
# can be used instead of pickle
from sklearn.externals import joblib

# More Cleaning Functions

from nltk.tokenize import word_tokenize

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


def tokenzer(text):
    token = word_tokenize(text)
    return token

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv('spam.csv', encoding='latin-1')

    # Features and labels
    df['label'] = df['v1'].map({'ham': 0, 'spam': 1}) # little preprocessing
    X = df['v2']
    y = df['label']

    # Extract Feature with CountVectorizer

    vectorizer = CountVectorizer(tokenizer=tokenzer, lowercase=True, ngram_range=(1,3))
    X = vectorizer.fit_transform(X)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

    # Using Naive bayes Classifier
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)

    # Using LogisticRegression Classifier
    
    logReg = LogisticRegression()
    logReg.fit(X_train, y_train)
    logReg.score(X_test, y_test)

     # Using SVM Classifier to handle class imbalance in data
    
    svc = SVC(class_weight='balanced', kernel='linear', C=0.01)
    svc.fit(X_train, y_train)
    svc.score(X_test, y_test)

    # Alternate Usage of Saved Models
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # NB_spam_model = open('NB_spam_model.pkl', 'rb')
    # clf = joblib.load(NB_spam_model)


    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = vectorizer.transform(data).toarray()
        my_prediction = logReg.predict(vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
