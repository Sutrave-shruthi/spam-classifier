import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    "spam.csv",
    sep='\t',
    names=['label', 'message'],
    encoding='latin-1'
)

# Convert labels
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Inputs and outputs
X = data['message']
y = data['label_num']

# Convert text into numbers
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Test custom message
msg = ["Congratulations! You won a free iPhone"]
msg_count = cv.transform(msg)

prediction = model.predict(msg_count)

if prediction[0] == 1:
    print("Spam Message")
else:
    print("Not Spam")