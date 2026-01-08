# AutoJudge-Predicting-Programming-Problem-Difficulty

🤖 AutoJudge: Programming Problem Difficulty Estimator
## 📌 What is AutoJudge?

AutoJudge is an AI-based tool that predicts the difficulty level of competitive programming problems.

It classifies problems into:

🟢 Easy

🟡 Medium

🔴 Hard

It also provides a numerical difficulty score, giving a more detailed idea of problem complexity.

🎯 Why AutoJudge?

Difficulty levels on coding platforms are often assigned manually and can be subjective.
AutoJudge helps by using machine learning to make difficulty prediction more consistent and automatic.

## 📊 Dataset Details

The dataset contains programming problems with the following fields:

 -Problem description

 -Input format

 -Output format

 -Constraints

 -Difficulty label (Easy / Medium / Hard)

 -Difficulty score

 ## 🧹 Data Preparation

Before training the models, the text data is cleaned and processed:

 -Text normalization

 -Stopword removal

 -Lemmatization

 -Class distribution analysis

 -Feature selection using Random Forest importance

# 🧠 How It Works
## 🧩 Feature Extraction

AutoJudge uses both text and numeric features:

 -TF-IDF for important words

 -Length of problem text

 -Count of mathematical symbols

 -Maximum constraint value

 -Detection of keywords like dp, graph, tree, greedy, etc.

## ⚙️ Models Used

#### 🧠 Classification

Random Forest
→ Predicts Easy / Medium / Hard

#### 📐 Regression

Linear Regression
→ Predicts difficulty score

Feature selection helped improve overall accuracy.

# 📈 Model Performance

-✅ Random Forest Accuracy: ~54%

-📉 MAE: ~2.5

-📉 RMSE: ~3.1

Better results were achieved after adding engineered features.

# 🖥️ Web App (Streamlit)

The project includes an interactive Streamlit web app.

### 🔄 User Flow

-✍️ Enter problem description

-⌨️ Enter input format

-📤 Enter output format

- Get instant predictions

### 🎨 UI Features

-✨ Glassmorphism design

-🌈 Gradient theme

-⏱️ Real-time results

-📊 Visual difficulty progress bar

## 🚀 Run the Project Locally
1️⃣ Clone the Repository

    git clone https://github.com/riteshiitr/AutoJudge-Predicting-Programming-Problem-Difficulty.git
    
    cd AutoJudge-Predicting-Programming-Problem-Difficulty

2️⃣ Install Dependencies

    pip install -r requirements.txt

3️⃣ Start the App

    streamlit run app_web.py

4️⃣ Open in Browser

    http://localhost:8501

# 🎥 Demo Video

▶️ Watch Demo:
👉 

# 👩‍💻 Author

👤 Name: Ritesh Kumar Ratnakar

🆔 Enrollment No: 23113128

🎓 Domain: Machine Learning, NLP, Web Deployment

🛠️ Tools: Python, Scikit-learn, NLTK, Streamlit
