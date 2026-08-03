# 🧬 PolySights: AI-Driven Polymer Analytical Engine

### Computational Property Modeling for Advanced Material Science

---

## 🎓 Academic Project Overview

* **Department:** Chemical Technology (CT)
* **Team Members:**

  * S. Arun Ulagappan
  * A. Vishaal
  * Akash S J


---

## 📝 Project Concept

**PolySights** is a predictive analytical tool designed to bridge the gap between molecular design and industrial manufacturing.

In traditional polymer science, determining key physical properties such as **Glass Transition Temperature (Tg)** and **Density** requires expensive laboratory instruments like Differential Scanning Calorimetry (DSC).

This project leverages **Machine Learning (Random Forest Regression)** to predict these properties instantly using only the polymer’s **chemical structure (SMILES string)**.

👉 It functions as a **Virtual Lab**, enabling chemical technologists to evaluate materials before synthesis.

---

## ✨ Key Features

* 🔬 **Molecular Fingerprinting**
  Converts chemical structures into numerical descriptors such as:

  * Molecular Weight (MolWt)
  * LogP (Hydrophobicity)
  * TPSA (Topological Polar Surface Area)
  * Rotatable Bonds

* 🏭 **Industrial Polymer Library**
  Includes pre-loaded high-performance polymers:

  * Kevlar
  * Teflon
  * PET
  * Polycarbonate
  * and more

* 🤖 **Automated Interpretation**
  Suggests real-world applications based on predicted properties (e.g., Aerospace, Medical, Packaging).

* 🎨 **Advanced UI**
  Professional dashboard with **Violet-to-White gradient** built using Streamlit.

---

## 🛠️ Tech Stack

| Component            | Technology Used         |
| -------------------- | ----------------------- |
| Programming Language | Python 3.9+             |
| Web Framework        | Streamlit               |
| Chemistry Engine     | RDKit                   |
| Machine Learning     | Scikit-Learn            |
| Data Handling        | Pandas, NumPy           |
| Visualization        | Pillow (Image Handling) |

---

## 🚀 Installation & Setup

### 1️⃣ Clone or Create Project

Create a new folder and save the main file as:

```
app.py
```

---

### 2️⃣ Install Dependencies

Run the following command in your terminal:

```bash
pip install streamlit rdkit pandas numpy scikit-learn pillow
```

---

### 3️⃣ Run the Application

```bash
streamlit run app.py
```

The application will automatically open in your default browser.

---

## 📖 Usage Guide (For Presentation)

### 🔹 Step 1: System Overview

* Navigate to **SYSTEM OVERVIEW** tab
* Explain theoretical background and KPIs

### 🔹 Step 2: Select Polymer

* Go to **ANALYSIS TERMINAL**
* Choose a polymer from **Industrial Examples**

### 🔹 Step 3: Run Prediction

* Click **RUN PREDICTIVE ANALYSIS**

### 🔹 Step 4: Analyze Results

* View **2D molecular structure**
* Observe predicted values:

  * Tg (Glass Transition Temperature)
  * Density
  * Molecular Weight
* Explain **Industrial Interpretation** output

---

## 📊 Methodology

### 1️⃣ Data Encoding

* Input polymer using **SMILES notation**

### 2️⃣ Feature Extraction (RDKit)

* **MolWt** → Molecular mass
* **LogP** → Hydrophobicity / chemical resistance
* **Rotatable Bonds** → Chain flexibility
* **TPSA** → Polar interaction capability

### 3️⃣ Machine Learning Model

* Algorithm: **Random Forest Regression**
* Predicts:

  * Glass Transition Temperature (Tg)
  * Density

---

## 📁 Project Structure

```
📂 Polymer-Predictor-Project
├── 📄 app.py              # Main Application Code
├── 📄 README.md           # Documentation
└── 📄 requirements.txt    # Dependencies
```

---

## 🎯 Applications

* Polymer Research & Development
* Material Screening before synthesis
* Industrial Design Optimization
* Academic Learning Tool

---

## ⚡ Future Enhancements

* Expand polymer dataset for higher accuracy
* Add more properties (Thermal Stability, Tensile Strength)
* Integrate Deep Learning models
* Cloud deployment for scalability

---

## 🏁 Conclusion

PolySights demonstrates how **AI + Cheminformatics** can significantly reduce the time, cost, and effort involved in polymer property evaluation.

It serves as a powerful **decision-support system** for chemical technologists, bridging theoretical modeling with practical industrial applications.

---

## 📜 License

This project is developed for academic purposes under the **CT Department Assessment - 2026**.

---
