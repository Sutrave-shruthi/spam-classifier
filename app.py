from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
data = pd.read_csv(
    "spam.csv",
    sep='\t',
    names=['label', 'message'],
    encoding='latin-1'
)

# Convert labels
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Prepare data
X = data['message']
y = data['label_num']

# Convert text to vectors
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X, y)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction page
@app.route('/predict', methods=['POST'])
def predict():

    msg = request.form['message']

    msg_vector = cv.transform([msg])

    prediction = model.predict(msg_vector)

    if prediction[0] == 1:
        result = "Spam Message"
    else:
        result = "Not Spam"

    return render_template(
        'index.html',
        prediction=result
    )

if __name__ == "__main__":
    app.run(debug=True)