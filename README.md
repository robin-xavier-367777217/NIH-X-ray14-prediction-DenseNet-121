# NIH Chest X-ray14 Disease Prediction

A deep learning project to predict 14 common thoracic pathologies from chest X-ray images using a DenseNet-121 model with transfer learning. This project was developed as part of my data science Up-Skilling.

---

## 🎯 Project Goal

The main objective was to build and train a multi-label classification model capable of identifying the presence of 14 different diseases from a single chest X-ray, based on the [NIH ChestX-ray14 dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community).

---

## 🚀 Technologies Used

- Python
- PyTorch & torchvision
- Pandas & scikit-learn
- Matplotlib & Seaborn
- Kaggle (for GPU training)

---

## 📊 Results

The model was trained in several stages, progressively unfreezing layers of the DenseNet architecture to fine-tune it on the X-ray data. The final performance on the test set, measured by the Area Under the Curve (AUC) for each pathology, is shown below.

![Model Performance Chart](./features/xray_predictor/metrics_images/new%20auc%20score.png)

---

## ⚙️ How to Use

Follow these steps to set up the project and run predictions.

### 1. Clone the Repository
```bash
git clone [https://github.com/robin-xavier-367777217/NIH-X-ray14-prediction-DenseNet-121.git](https://github.com/robin-xavier-367777217/NIH-X-ray14-prediction-DenseNet-121.git)
cd NIH-X-ray14-prediction-DenseNet-121
```

### 2. Install Dependencies
It's highly recommended to use the provided `requirements.txt` file to ensure a consistent environment.
```bash
pip install -r requirements.txt
```

### 3. Run the Prediction Script
Provide the path to an X-ray image to get a prediction.
```bash
python -m features.xray_predictor.predict /path/to/your/xray_image.jpeg
```

---

## 🔗 Connect with Me

Feel free to reach out or check out my other work!

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/robin-xavier-367777217/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/robin-xavier-367777217)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/robinxavier4kaggle)

---

## 🙏 Acknowledgements

- This project utilizes the NIH ChestX-ray14 dataset.
- I have written a detailed article about the complete project workflow on LinkedIn. You can read it here: **[https://www.linkedin.com/in/robin-xavier-367777217/]**

---

## ⚠️ Medical Disclaimer

This project is for **educational and research purposes only**. The model is an academic exploration and **is not a substitute for professional medical advice, diagnosis, or treatment.**

**Do not use the information or predictions from this model to make any health-related decisions.** Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have seen or read in this repository.

The author is not responsible or liable for any advice, course of treatment, diagnosis, or any other information, services, or products that you obtain through this project.
