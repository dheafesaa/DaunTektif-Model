# 🌽 Maize Leaf Disease Classification API

This project is a **FastAPI-based REST API** for classifying maize leaf diseases using a fine-tuned **MobileNetV3-Small** model.  
It predicts one of the following classes:

- **Maize Leaf Blight**  
- **Maize Leaf Spot**  
- **Maize Streak Virus**

For each prediction, the API also returns a description, prevention tips, and treatment suggestions.

---

## 📦 Requirements

- Python 3.9+ (recommended)
- Virtual environment (venv/conda)
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Pillow
- NumPy

---

## ⚙️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/maize-disease-api.git
   cd maize-disease-api
2. **Create virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows
3. **Install dependencies**  
   ```bash
   pip install

---

## ▶️ Running the API
- **Start the server with**  
   ```bash
   python main.py
- **or directly with Uvicorn:**  
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

The API will run at: http://127.0.0.1:8000

---

## 📤 Usage

### Endpoint: `/predict`

- **Method:** `POST`  
- **Content-Type:** `multipart/form-data`  
- **Body Parameter:**  
  - `file`: an image file of a maize leaf (e.g., `.jpg`, `.png`)

---

### Example with `curl`

- ```bash
  curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@sample_leaf.jpg"

---

## 📜 License

© 2025 **DaunTektif** 🔍🌱 All Rights Reserved 

This code is the **intellectual property** of **DaunTektif**.  

✅ Permitted use:  
- Educational and research purposes 📚🔬  
- Personal learning and experimentation 🧑‍💻  

❌ Prohibited use:  
- Commercial usage without prior written permission 💰🚫  
- Redistribution, reproduction, or modification for profit 🛑  

---

## 👨‍💻 Author & Contact

Developed with ❤️ by **DaunTektif**  

📧 Email: [depsaaa0823@gmail.com]  
