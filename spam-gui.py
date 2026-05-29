import pandas as pd
from tkinter import *
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------------------
# LOAD DATASET
# -------------------------------

data = pd.read_csv(
    "spam.csv",
    sep='\t',
    names=['label', 'message'],
    encoding='latin-1'
)

# Convert labels into numbers
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

# Inputs and outputs
X = data['message']
y = data['label_num']

# Convert text into vectors
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------------------
# GUI FUNCTION
# -------------------------------

def check_message():
    msg = entry.get("1.0", END).strip()

    if msg == "":
        messagebox.showwarning("Warning", "Please enter a message")
        return

    # Convert message into vector
    msg_vector = cv.transform([msg])

    # Predict
    prediction = model.predict(msg_vector)

    # Display result
    if prediction[0] == 1:
        result_label.config(text="Spam Message", fg="red")
    else:
        result_label.config(text="Not Spam", fg="green")


# -------------------------------
# TKINTER WINDOW
# -------------------------------

root = Tk()
root.title("Spam Message Detector")
root.geometry("500x400")
root.config(bg="lightblue")

# Heading
title = Label(
    root,
    text="AI Spam Message Detector",
    font=("Arial", 20, "bold"),
    bg="lightblue"
)
title.pack(pady=20)

# Text box
entry = Text(root, height=8, width=50, font=("Arial", 12))
entry.pack(pady=10)

# Button
check_btn = Button(
    root,
    text="Check Message",
    font=("Arial", 14),
    command=check_message
)
check_btn.pack(pady=10)

# Result label
result_label = Label(
    root,
    text="",
    font=("Arial", 18, "bold"),
    bg="lightblue"
)
result_label.pack(pady=20)

# Run app
root.mainloop()