from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import numpy as np
import uvicorn

app = FastAPI()
model = load_model('MobileNetV3-Small_finetune_RMSprop.h5')

CLASS_NAMES = ["Maize Leaf Blight", "Maize Leaf Spot", "Maize Streak Virus"]

COPYWRITING = {
    "Maize Leaf Blight": {
        "description": (
            "Maize Leaf Blight adalah penyakit daun jagung yang ditandai dengan bercak cokelat memanjang di bagian tengah hingga ujung daun dan biasanya berawal dari daun bawah lalu menyebar ke daun atas akibat infeksi jamur Helminthosporium turcicum."
        ),
        "prevention": (
            "Tanam jagung dengan jarak yang cukup agar udara bisa mengalir",
            "Siram tanah di sekitar tanaman tanpa membasahi daun secara langsung",
            "Periksa kondisi tanaman setiap minggu untuk mendeteksi gejala lebih awal",
        ),
        "treatment": (
            "Pangkas dan buang daun yang terinfeksi untuk mencegah penyebaran",
            "Bersihkan sisa daun jatuh dan gulma secara teratur di sekitar tanaman",
            "Semprotkan fungisida alami jika bercak terus bertambah banyak pada daun",
        ),
    },
    "Maize Leaf Spot": {
        "description": (
            "Maize Leaf Spot merupakan penyakit yang ditandai dengan bintik-bintik kecil berwarna cokelat atau kehitaman pada permukaan daun jagung. Bercak ini awalnya tersebar acak dan lama-kelamaan dapat menyatu membentuk pola yang lebih luas di permukaan daun."
        ),
        "prevention": (
            "Pastikan tanaman jagung mendapatkan cukup sinar matahari",
            "Atur sistem pengairan supaya tanah tidak terlalu becek atau lembap",
            "Ganti tanaman jagung dengan jenis lain pada musim berikutnya untuk mengurangi risiko penyakit"
        ),
        "treatment": (
            "Potong bagian daun yang ada bercak lalu bakar agar penyakit tidak menular",
            "Kumpulkan dan buang semua sisa tanaman dari lahan setiap selesai panen",
            "Semprot daun dengan fungisida organik jika bercak mulai menyebar ke daun lain"
        ),
    },
     "Maize Streak Virus": {
        "description": (
            "Maize Streak Virus adalah penyakit pada jagung yang ditandai dengan munculnya garis-garis kuning atau putih memanjang sejajar tulang daun. Daun yang terinfeksi tampak kaku, kerdil, dan sering mengalami perubahan warna pada permukaannya."
        ),
        "prevention": (
            "Tanam jagung di awal musim tanam agar risiko serangan serangga lebih rendah",
            "Bersihkan alat pertanian setelah digunakan supaya virus tidak menyebar ke lahan lain",
            "Amati pertumbuhan daun jagung setiap minggu agar bisa langsung bertindak saat muncul gejala"
        ),
        "treatment": (
            "Cabut tanaman yang sudah parah terkena virus agar tidak menular ke tanaman lain",
            "Semprot kutu daun dengan larutan sabun cair agar populasinya tetap terkendali",
            "Singkirkan gulma di sekitar jagung supaya tidak jadi tempat tinggal serangga pembawa virus"
        ),
    },
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"==> [DEBUG] File received: {file.filename}", flush=True)
    contents = await file.read()
    print(f"==> [DEBUG] File size: {len(contents)} bytes", flush=True)
    file.file.seek(0)
    img = Image.open(file.file).convert("RGB").resize((224, 224))
    arr = np.asarray(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    pred = model.predict(arr)[0]
    idx = int(np.argmax(pred))
    pred_class = CLASS_NAMES[idx]
    confidence = float(pred[idx]) * 100  

    for i, class_name in enumerate(CLASS_NAMES):
        print(f"==> [CONFIDENCE] {class_name}: {pred[i]:.4f}", flush=True)
    print(f"==> [DEBUG] Predicted class: {pred_class} ({confidence:.2f}%)", flush=True)

    return {
        "class": pred_class,
        "confidence": f"{confidence:.2f}%",  
        "description": COPYWRITING[pred_class]["description"],
        "prevention": COPYWRITING[pred_class]["prevention"],
        "treatment": COPYWRITING[pred_class]["treatment"],
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
