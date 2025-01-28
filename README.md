## **🫀 Heart Disease Prediction App**

### **Overview**
This Streamlit-based application predicts the likelihood of heart disease based on user-provided medical parameters. The app utilizes a Random Forest Classifier model and various data preprocessing techniques to provide accurate predictions.

---

### **Key Features**
- User-friendly input fields for medical parameters like age, blood pressure, cholesterol levels, and more.
- Automatic preprocessing and feature encoding for categorical data.
- Provides predictions indicating the presence or absence of heart disease.

---

### **Tech Stack**
- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy

---

### **How It Works**
1. Users input their health data, including parameters such as age, chest pain type, cholesterol levels, and heart rate.
2. The app preprocesses the input data:
   - Categorical variables are label-encoded using pre-trained encoders.
   - Data is scaled using a pre-saved scaler (`Scaler.pkl`).
3. The Random Forest Classifier (`random_forest_model.pkl`) is used to predict whether heart disease is likely.
4. Results are displayed in a user-friendly format.

---

### **Installation Instructions**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Heart-Disease-Prediction
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run Heart_app.py
   ```

---

### **Usage**
- Open the app and input the required health parameters.
- Click the "Predict" button to get the results.

---

### **Model Information**
- **Model:** Random Forest Classifier
- **Preprocessing:** Label encoding and scaling
- **Files Required:**  
  - `random_forest_model.pkl`: Pre-trained model file  
  - `Scaler.pkl`: Scaler for data normalization  
  - `features.csv`: Feature list for maintaining data consistency  

---

### **Future Improvements**
- Expand the input parameters for a more comprehensive prediction.
- Add data visualization for better insights.
- Implement additional machine learning models for comparison.

---

### **Contributors**
- [Your Name]

---

### **License**
MIT License. 

---

Would you like any refinements to this README as well? 
