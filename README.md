# 🧠 Loan Risk Assessment using Neural Network

This project implements a neural network **from scratch (no ML libraries)** to assess the **risk of loan default** based on historical financial and personal data. The model is built using Python with custom code for data preprocessing, forward propagation, backpropagation, and weight updates.

---

## 📌 Project Highlights

- 🧮 **Binary Classification** – Predict if a loan will be repaid or defaulted.
- 🔧 **Neural Network from Scratch** – No TensorFlow or PyTorch used.
- 🔍 **Accuracy up to 84.94%** after tuning hidden layers and learning rates.
- 💡 **Real-time prediction support** based on user input.
- 📊 **Kaggle loan dataset** with 8,145 records.

---

## 🧰 Tech Stack

- **Language**: Python  
- **Libraries**: `pandas`, `numpy`, `math`, `random`  
- **Model**: Manual Neural Network (Sigmoid activation, MSE loss)  
- **Dataset Source**: Kaggle (Loan Default Prediction Dataset)

---

## 🧪 Features Used

- Age  
- Income  
- Home Ownership  
- Loan Amount  
- Employment Length  
- Loan Intent (purpose)  
- Interest Rate  
- Percent Income  
- Credit Length  
- Default History

---

## ⚙️ How It Works

1. **Data Preprocessing**
   - Categorical variables encoded.
   - Missing values handled.
   - Features normalized between 0 and 1.

2. **Neural Network Architecture**
   - Input Layer → Hidden Layers → Output Layer
   - Sigmoid activation in all layers.
   - Loss function: Mean Squared Error (MSE)

3. **Training & Evaluation**
   - Training/testing split: 80/20
   - Learning rate tuning
   - Epochs: 500 or more
   - Accuracy measured on test data

4. **Real-Time Prediction**
   - User inputs new applicant data
   - Model predicts the loan status instantly

---

## 📈 Best Accuracy Achieved

| Case | Hidden Layers | Learning Rate | Epochs | Accuracy |
|------|----------------|----------------|--------|-----------|
| 8    | 2 (10+30+1)     | 0.01           | 500    | **84.94%** |

---

## 🚀 Future Enhancements

- Add GUI for user-friendly predictions (Tkinter or Web-based)
- Compare with logistic regression and sklearn models
- Visualize training loss and accuracy trends

---

## 🧑‍💻 Developed by

**P. Anbu Selvan**  
B.Sc. Computer Science – VHNSN College, Madurai Kamaraj University  
[LinkedIn](https://linkedin.com/in/anbuselvan) | [GitHub](https://github.com/Anbu0396)

---

## 📄 License

This project is for academic use and learning purposes only.
