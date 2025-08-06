
# 🚗 FIUS Based Infant Detection in Car Seat

This project uses ultrasonic sensors and machine learning to detect:

* Whether a **baby carrier** is on a car seat
* Whether a **baby is present** in that carrier
* Even when the baby is **covered by a blanket or sunscreen**

Built as part of the **Autonomous Intelligent Systems and Machine Learning** module at **Frankfurt University of Applied Sciences**, this system enhances in-vehicle safety through signal processing and intelligent classification models.

---

## 🧠 Project Overview

Traditional sensors like cameras face challenges in low light or privacy-sensitive environments. This project proposes a non-visual, ultrasonic-based approach using:

* **Red Pitaya + FIUS ultrasonic sensor**
* **Signal processing (FFT)**
* **Machine learning classifiers**

---

## 🧪 Key Features

* 🚗 Detects **presence of infant car seat** and **baby**
* 🧺 Works even with **blanket or sunscreen obstructions**
* 📊 Real-time data collection using Red Pitaya + SRF02 sensors
* 🔍 Signal preprocessing using FFT
* 🧠 ML models: XGBoost, MLP, Random Forest, SVM
* 📈 Evaluated with confusion matrices and classification reports

---

## 🏗️ Architecture

### 📡 Hardware

* **Red Pitaya** measurement board
* **FIUS Ultrasonic Sensor** (SRF02)
* Baby dolls used to simulate infant presence
* Test environment: *Ford Fiesta v16* in controlled conditions

### 💻 Software & Tools

* Python (NumPy, Scikit-learn, Pandas, XGBoost)
* GUI tool for Red Pitaya data acquisition via UDP
* Manual ADC to FFT conversion
* Feature engineering: spectral + phase features

---

## ⚙️ ML Tasks & Models

| Task  | Description                         | Models Used        |
| ----- | ----------------------------------- | ------------------ |
| **1** | Detect baby carrier on seat         | XGBoost, MLP       |
| **2** | Detect baby in carrier              | Random Forest, SVM |
| **3** | Detect baby under blanket/sunscreen | XGBoost            |

### Key Preprocessing Steps:

* Raw ADC data (1.953 MHz sampling) → FFT
* Feature extraction (mean, entropy, centroid, etc.)
* Data augmentation with perturbation (+/- 3%)
* Feature selection to address overfitting

---

## 📊 Results Summary

| Model             | Raw Data Accuracy | After Feature Extraction | Final Accuracy (Feature Selected) |
| ----------------- | ----------------- | ------------------------ | --------------------------------- |
| **XGBoost**       | \~54%             | 100% (overfit)           | **91.58%**                        |
| **MLP**           | \~55%             | 100% (overfit)           | **90.42%**                        |
| **Random Forest** | \~57%             | 100% (overfit)           | Improved after selection          |

✅ Feature selection using correlation matrix improved generalization
📉 Addressed overfitting by dropping redundant features

---

## 📂 Dataset Summary

* **Total samples:** 129,000
* **Tasks:**

  * Task 1: 50,000 samples
  * Task 2: 46,000 samples
  * Task 3: 33,000 samples
* **Data split:** 80% training / 20% testing

---

## 📷 Visuals

* Sensor mounted at 28° angle on dashboard
* Baby dolls in seated, lying, strapped positions
* FFT spectra and classification labels
* Confusion matrices and ROC curves

---

## 👨‍💻 Authors

* **Anushruthpal Keshavathi Jayapal**
* **Aswini Thirumaran**
* **Lavanya Suresh**
* **Mandar Gokul Kale**

---

## 📜 License

This project is for academic and research purposes. Contact authors for reuse or collaboration.

---

## 📧 Contact

For inquiries or collaboration:

* [mandar.kale@stud.fra-uas.de](mailto:mandar.kale@stud.fra-uas.de)
* [aswini.thirumaran@stud.fra-uas.de](mailto:aswini.thirumaran@stud.fra-uas.de)
* [anushruthpal.keshavathi-jayapal@stud.fra-uas.de](mailto:anushruthpal.keshavathi-jayapal@stud.fra-uas.de)
* [lavanya.suresh@stud.fra-uas.de](mailto:lavanya.suresh@stud.fra-uas.de)

