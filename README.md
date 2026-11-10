# 🌊 Rising Waters – AI-Based Flood Prediction System

## 📌 Project Overview

Rising Waters is an AI-powered Flood Prediction System developed using Machine Learning and Flask. The application predicts whether an area is at **High Flood Risk** or **Low Flood Risk** based on environmental and geographical factors. The system uses an XGBoost classifier trained on historical flood-related data to provide accurate flood risk predictions along with confidence scores.

---

## 🎯 Objectives

- Predict flood risk using machine learning.
- Help users understand the possibility of flooding based on environmental conditions.
- Provide a simple and user-friendly web interface.
- Maintain prediction history for analysis.

---

## ✨ Features

- 🌧️ Flood Risk Prediction
- 🤖 XGBoost Machine Learning Model
- 📊 Prediction Confidence Score
- 📈 Prediction History
- 💻 User-Friendly Web Interface
- 📱 Responsive Design

---

## 🛠️ Technologies Used

### Programming Language
- Python

### Machine Learning
- XGBoost
- Scikit-learn

### Web Framework
- Flask

### Frontend
- HTML
- CSS
- Bootstrap

### Libraries
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```
Rising-Waters/
│
├── data/
│   └── prediction_history.csv
│
├── datasets/
│   └── flood.csv
│
├── models/
│   ├── floods.save
│   └── scaler.save
│
├── notebook/
│   └── FloodPrediction.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   ├── about.html
│   ├── result.html
│   └── history.html
│
├── app.py
├── requirements.txt
├── Project Report.pdf
├── Demo.mp4
└── README.md
```

---

## 📊 Machine Learning Workflow

1. Dataset Collection
2. Data Preprocessing
3. Feature Selection
4. Data Scaling
5. Model Training
6. Model Evaluation
7. Flood Prediction
8. Web Application Deployment

---

## 📈 Model Used

**XGBoost Classifier**

The model predicts flood risk using the following five important features:

- Monsoon Intensity
- Drainage Systems
- River Management
- Deforestation
- Urbanization

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/SGK5357k/Rising-Waters.git
```

### Navigate to the Project Folder

```bash
cd Rising-Waters
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

---

## 🌐 Application

After running the application, open:

```
http://127.0.0.1:5000
```

---

## 📸 Application Screens

- Home Page
- About Page
- Prediction Form
- Prediction Result
- Prediction History

---

## 📄 Project Report

The complete project documentation is available in:

```
Project Report.pdf
```

---

## 🎥 Demo Video

The complete demonstration of the project is available in:

```
Demo.mp4
```

---

## 📋 Sample Input

| Feature | Example Value |
|----------|--------------:|
| Monsoon Intensity | 8 |
| Drainage Systems | 3 |
| River Management | 2 |
| Deforestation | 8 |
| Urbanization | 7 |

---

## 📌 Output

The application predicts:

- ✅ Low Flood Risk
- ⚠️ High Flood Risk
- Prediction Confidence (%)

---

## 🔮 Future Enhancements

- Real-time weather API integration
- Interactive flood risk maps
- SMS and Email alert system
- Mobile application
- IoT sensor integration
- Cloud deployment

---

## 📚 References

- Python Documentation
- Flask Documentation
- Scikit-learn Documentation
- XGBoost Documentation
- Pandas Documentation
- NumPy Documentation

---

## 👨‍💻 Developed By

**K. Sai Gopala Krishna**

B.Tech – Computer Science and Engineering (Artificial Intelligence & Machine Learning)

---

## 📜 License

This project is developed for educational and internship purposes.