#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import PyPDF2
import nltk
import re
import joblib
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Function to clean text
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove numbers/symbols
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Load dataset
df = pd.read_csv('Resume.csv')
df['cleaned_text'] = df['Resume_str'].apply(preprocess_text)


# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        print(f"Error reading {pdf_path}: {str(e)}")
    return preprocess_text(text)

# Load PDFs from directory
pdf_folder = "data"  # Change this to your actual folder path
pdf_texts = []
pdf_categories = []

for category in os.listdir(pdf_folder):  # Assuming folders are named by category
    category_path = os.path.join(pdf_folder, category)
    if os.path.isdir(category_path):
        for pdf_file in os.listdir(category_path):
            pdf_path = os.path.join(category_path, pdf_file)
            text = extract_text_from_pdf(pdf_path)
            if text:
                pdf_texts.append(text)
                pdf_categories.append(category)

# Convert to DataFrame and merge with CSV data
pdf_df = pd.DataFrame({"cleaned_text": pdf_texts, "Category": pdf_categories})

df = pd.concat([df[['cleaned_text', 'Category']], pdf_df], ignore_index=True)


# In[2]:


pip install seaborn


# In[3]:


# Display the first few rows of the DataFrame to see how the data looks
print(df.head())


# In[4]:


# Show some basic info about the dataset
print(df.info())


# In[5]:


# Check the distribution of categories
print(df['Category'].value_counts())

# Show the first few cleaned text entries
print(df['cleaned_text'].head())


# In[6]:


# View some samples of cleaned text
for i in range(5):
    print(f"Sample {i+1} cleaned text: {df['cleaned_text'].iloc[i]}")


# In[7]:


# Show a random sample of rows
print(df.sample(5))


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

# Data Visualization
def plot_category_distribution():
    plt.figure(figsize=(10, 5))
    sns.countplot(y=df['Category'], order=df['Category'].value_counts().index, palette='coolwarm')
    plt.title("Distribution of Resume Categories")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.show()




# Display visualizations
plot_category_distribution()


# In[9]:


from collections import Counter
from nltk.tokenize import word_tokenize

# Tokenize the cleaned text and get the frequency of each word
all_words = ' '.join(df['cleaned_text']).split()
word_freq = Counter(all_words)

# Get the top 10 most common words
top_words = word_freq.most_common(10)

# Plot the top 10 words
top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

plt.figure(figsize=(10, 6))
sns.barplot(x='Frequency', y='Word', data=top_words_df, palette='Blues_d')
plt.title('Top 10 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()


# In[10]:


pip install imbalanced-learn


# In[11]:


from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


# Encode target labels
label_encoder = LabelEncoder()
df['Category_encoded']=label_encoder.fit_transform(df['Category'])

# Convert text into numerical format
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))  # Unigrams + Bigrams
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['Category_encoded']

# Apply SMOTE to balance dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=150, random_state=42, class_weight="balanced"),
    'LogisticRegression': LogisticRegression(max_iter=700),
    'SVM': SVC(kernel='linear', probability=True),
    'XGBoost': XGBClassifier(eval_metric='mlogloss')
}

best_model, best_accuracy = None, 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔹 {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=1))
    
    if acc > best_accuracy:
        best_accuracy, best_model = acc, model

# Save the best model and vectorizer
joblib.dump(best_model, 'best_resume_classifier1.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer1.pkl')
joblib.dump(label_encoder, 'label_encoder1.pkl')

print("\n✅ Model training complete. Best model saved!")


# In[12]:


pip install pymupdf


# In[ ]:


import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import joblib
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
from pathlib import Path

# Load trained model, vectorizer, and label encoder
model = joblib.load('best_resume_classifier1.pkl')
vectorizer = joblib.load('tfidf_vectorizer1.pkl')
label_encoder = joblib.load('label_encoder1.pkl')

# Create the Categorized_Resumes folder if it doesn't exist
categorized_resumes_path = "Categorized_Resumes"
os.makedirs(categorized_resumes_path, exist_ok=True)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text")  # Extract plain text from each page
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"❌ Error extracting text from {pdf_path}: {e}")
        return ""

# Function to classify uploaded resume
def classify_resume():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if not file_path:
        return

    # Extract text from the PDF file
    text = extract_text_from_pdf(file_path)
    if not text:
        messagebox.showerror("Error", "Failed to extract text from PDF. Try another file.")
        return

    print("\n📝 Extracted Text Preview:\n", text[:1000])  # Print first 1000 characters for debugging

    # Transform text using TF-IDF
    vectorized_text = vectorizer.transform([text]).toarray()

    # Predict category & confidence
    probabilities = model.predict_proba(vectorized_text)[0]
    encoded_category = np.argmax(probabilities)
    confidence_score = max(probabilities)

    # Decode category
    category = label_encoder.inverse_transform([encoded_category])[0]

    print(f"🔹 Predicted: {category} | Confidence: {confidence_score:.2f}\n")

    # Move to category folder
    category_folder = os.path.join(categorized_resumes_path, category)
    os.makedirs(category_folder, exist_ok=True)
    shutil.move(file_path, os.path.join(category_folder, os.path.basename(file_path)))

    messagebox.showinfo("Success", f"Resume classified as {category} (Confidence: {confidence_score:.2f})")
    update_category_list()

# Function to update the list of available categories in the GUI
def update_category_list():
    listbox_categories.delete(0, tk.END)

    # Check the Categorized_Resumes folder for subfolders (categories)
    if os.path.exists(categorized_resumes_path):
        categories = [f.name for f in Path(categorized_resumes_path).iterdir() if f.is_dir()]
    else:
        categories = []
    
    # Add categories to Listbox
    for category in categories:
        listbox_categories.insert(tk.END, category)

# Function to open the selected folder
def open_category_folder():
    selected_category = listbox_categories.get(tk.ACTIVE)
    if selected_category:
        folder_path = os.path.join(categorized_resumes_path, selected_category)
        if os.path.exists(folder_path):
            os.startfile(folder_path) if os.name == 'nt' else os.system(f'xdg-open "{folder_path}"')
        else:
            messagebox.showerror("Error", "The folder could not be opened!")
    else:
        messagebox.showwarning("No Category Selected", "Please select a category to open.")

# Create the main GUI window
root = tk.Tk()
root.title("Resume Classifier")
root.geometry("500x500")
root.config(bg="#f4f4f9")  # Light background color

# Create a frame for buttons
frame_buttons = ttk.Frame(root, padding="20")
frame_buttons.pack(pady=20, fill="x")

# Upload Resume Button
btn_upload = ttk.Button(frame_buttons, text="Upload Resume", command=classify_resume, width=20)
btn_upload.grid(row=0, column=0, padx=10, pady=10)

# View Categorized Resumes Button
btn_view = ttk.Button(frame_buttons, text="View Categorized Resumes", command=open_category_folder, width=20)
btn_view.grid(row=0, column=1, padx=10, pady=10)

# Create a frame for Listbox and scrollbar
frame_listbox = ttk.Frame(root, padding="10")
frame_listbox.pack(fill="both", expand=True)

# Listbox to display categories
listbox_categories = tk.Listbox(frame_listbox, height=10, width=40, font=("Arial", 12), selectmode=tk.SINGLE)
listbox_categories.pack(side="left", fill="both", expand=True)

# Scrollbar for Listbox
scrollbar = ttk.Scrollbar(frame_listbox, orient="vertical", command=listbox_categories.yview)
scrollbar.pack(side="right", fill="y")
listbox_categories.config(yscrollcommand=scrollbar.set)

# Update the category list after initializing the GUI
update_category_list()

# Run the main loop
root.mainloop()

