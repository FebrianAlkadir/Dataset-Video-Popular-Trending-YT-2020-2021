# Prediksi Harga Rumah

Folder ini berisi proyek machine learning untuk memperkirakan harga jual rumah (`SalePrice`) menggunakan dataset Ames Housing. Notebook menjelaskan proses eksplorasi data, pembersihan, visualisasi, pemodelan, evaluasi, dan deployment API sederhana dengan Flask.

## Isi Folder

- `Houseprice.ipynb` - notebook pembelajaran dari eksplorasi hingga pengujian endpoint.
- `train.csv` dan `test.csv` - data training berlabel dan data fitur untuk pengujian.
- `data_description.txt` - keterangan fitur dataset.
- `sample_submission.csv` - contoh format file submission.
- `data.json` - contoh input API berupa vektor 76 nilai numerik.
- `train_model.py` - preprocessing, perbandingan model, evaluasi, dan penyimpanan model.
- `house_price_model.joblib` - model dan metadata preprocessing yang digunakan API.
- `testing_deploy.py` - server Flask untuk prediksi.
- `test_deployment.py` - tes format input dan endpoint.

## Evaluasi Model

Empat pendekatan dibandingkan: baseline median, Ridge Regression, Random Forest, dan Gradient Boosting. Pemilihan model menggunakan rata-rata RMSE dari 3-fold cross-validation pada bagian training. Evaluasi akhir dilakukan pada holdout 20%.

Gradient Boosting memperoleh RMSE cross-validation terendah dan dipilih sebagai model. Hasil holdout yang tercatat:

| Metrik | Hasil | Makna singkat |
|---|---:|---|
| MAE | 11,976 | Rata-rata besar selisih absolut prediksi. |
| MSE | 271,811,265 | Rata-rata kuadrat error; lebih sensitif terhadap error besar. |
| RMSE | 16,487 | Akar MSE, dalam satuan target harga. |
| R-squared | 0.889 | Sekitar 88.9% variasi target pada data holdout dijelaskan model. |

Nilai MAE, MSE, dan RMSE memakai skala `SalePrice` asli. R-squared menunjukkan kecocokan pada data holdout dan bukan persentase prediksi yang pasti benar.

## Menjalankan Proyek

Gunakan Python 3.13 atau lingkungan Python yang kompatibel. Buka Command Prompt atau terminal, lalu pindah ke folder ini:

```cmd
cd /d "D:\path\to\Dataset Houseprice"
```

Pasang dependensi jika belum tersedia:

```cmd
python -m pip install numpy pandas scikit-learn joblib flask
```

### Train dan evaluasi ulang

Jalankan:

```cmd
python train_model.py
```

Script akan menampilkan skor cross-validation dan metrik holdout, lalu membuat atau memperbarui `house_price_model.joblib`.

### Tes otomatis deployment

Jalankan:

```cmd
python -m unittest -v test_deployment.py
```

### Menjalankan API

Di terminal pertama, jalankan server:

```cmd
python testing_deploy.py
```

Biarkan proses tersebut berjalan. Di terminal kedua, pindah ke folder yang sama dan kirim contoh `data.json`:

```cmd
curl.exe -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary "@data.json"
```

Respons berisi array prediksi, misalnya:

```json
{"prediction": [0.5415]}
```

Input harus berbentuk objek JSON dengan kunci `data`; setiap elemen di dalamnya berupa array 76 angka, sesuai preprocessing pada materi. Output adalah `SalePrice` terstandardisasi seperti contoh materi.

> Flask development server ditujukan untuk pengujian lokal, bukan deployment produksi.
