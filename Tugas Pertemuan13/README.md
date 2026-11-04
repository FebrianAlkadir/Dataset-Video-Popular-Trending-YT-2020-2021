# Tugas Pertemuan 13 — Jenis-Jenis Klasifikasi

Folder ini berisi sembilan dataset untuk latihan identifikasi jenis klasifikasi. Dataset dibagi menjadi tiga kelompok. Setiap kelompok memiliki satu contoh klasifikasi biner, satu multikelas, dan satu multilabel. Semua CSV berisi setidaknya 150 baris. Notebook menggunakan pandas untuk membaca dataset, menghitung jumlah baris dan fitur, serta menampilkan distribusi label. Untuk dataset multilabel, notebook juga menghitung jumlah label aktif pada setiap sampel.

Materi acuan: [Notion — Jenis-Jenis Klasifikasi](https://app.notion.com/p/Jenis-Jenis-Klasifikasi-1-405e27a5dc7b80708a7aff22968f64cf).

## Struktur folder

Setiap `Dataset 1`, `Dataset 2`, dan `Dataset 3` berisi tiga berkas CSV dan satu notebook bernama `analisis.ipynb`. Buka notebook dari foldernya masing-masing dan jalankan sel secara berurutan. Pastikan Python, Jupyter, dan pandas tersedia.

| Kelompok | Jenis klasifikasi | Dataset | Baris | Fitur | Target |
|---|---|---|---:|---:|---|
| Dataset 1 | Biner | Wisconsin Diagnostic Breast Cancer | 569 | 30 | `diagnosis`: B atau M |
| Dataset 1 | Multikelas | Wine Recognition | 178 | 13 | `class`: 1, 2, atau 3 |
| Dataset 1 | Multilabel | Emotions | 593 | 72 | 6 label emosi |
| Dataset 2 | Biner | Banknote Authentication | 1.372 | 4 | `Class`: 1 atau 2 |
| Dataset 2 | Multikelas | Iris | 150 | 4 | `class`: 3 spesies |
| Dataset 2 | Multilabel | Flags | 194 | 19 | 7 label warna bendera |
| Dataset 3 | Biner | Pima Indians Diabetes | 768 | 8 | `class`: positif atau negatif |
| Dataset 3 | Multikelas | Glass Identification | 214 | 9 | `Type`: 6 jenis kaca |
| Dataset 3 | Multilabel | Yeast Protein Localization | 2.417 | 103 | 14 label lokasi protein |

## Ringkasan jenis klasifikasi

- **Biner:** setiap baris masuk tepat ke satu dari dua kategori.
- **Multikelas:** setiap baris masuk tepat ke satu kategori, dan jumlah kategorinya lebih dari dua.
- **Multilabel:** setiap baris dapat memiliki lebih dari satu label aktif sekaligus. Nilai target tiap label menunjukkan apakah label tersebut aktif (1) atau tidak (0).

## Keterangan dan sumber dataset

### Dataset 1

1. **Wisconsin Diagnostic Breast Cancer — biner.** Pengukuran inti sel digunakan untuk mengidentifikasi tumor jinak (`B`) atau ganas (`M`).
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>
   - Data asli: <https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data>
2. **Wine Recognition — multikelas.** Tiga belas hasil pengukuran kimia digunakan untuk memprediksi satu dari tiga kelas wine.
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/109/wine>
   - Data asli: <https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data>
3. **Emotions — multilabel.** Fitur akustik lagu dipasangkan dengan enam label emosi. Satu lagu dapat memiliki beberapa label. CSV menggabungkan partisi train1 dan test1 dari repositori ExecuteMulan.
   - Repositori data: <https://github.com/kdis-lab/ExecuteMulan/tree/f0c9bf5de82b60cc670c8f07eb16927eda74b66e/data/emotions>
   - [Emotions train1](https://raw.githubusercontent.com/kdis-lab/ExecuteMulan/f0c9bf5de82b60cc670c8f07eb16927eda74b66e/data/emotions/emotions-train1.arff) dan [Emotions test1](https://raw.githubusercontent.com/kdis-lab/ExecuteMulan/f0c9bf5de82b60cc670c8f07eb16927eda74b66e/data/emotions/emotions-test1.arff)

### Dataset 2

1. **Banknote Authentication — biner.** Empat fitur dari citra uang kertas digunakan untuk membedakan dua kategori keaslian.
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/267/banknote+authentication>
   - CSV yang digunakan: <https://raw.githubusercontent.com/datasets/openml-datasets/ac945980fbb83bf4200a893593a6c1099679d150/data/banknote-authentication/banknote-authentication.csv>
2. **Iris — multikelas.** Empat ukuran bunga digunakan untuk memprediksi satu dari tiga spesies Iris.
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/53/iris>
   - CSV yang digunakan: <https://raw.githubusercontent.com/datasets/openml-datasets/ac945980fbb83bf4200a893593a6c1099679d150/data/iris/iris.csv>
3. **Flags — multilabel.** Data negara memiliki tujuh target warna. Sebuah bendera dapat memuat beberapa warna sekaligus; versi multilabel berasal dari koleksi Mulan.
   - Halaman dataset UCI: <https://archive.ics.uci.edu/dataset/40/flags>
   - Berkas multilabel Mulan: <https://github.com/tsoumakas/mulan/blob/43f6389095ad71984264db4387a8074b22bc7d4a/data/multi-label/flags/flags.arff>

### Dataset 3

1. **Pima Indians Diabetes — biner.** Delapan fitur klinis digunakan untuk memprediksi hasil tes diabetes positif atau negatif.
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/34/diabetes>
   - CSV yang digunakan: <https://raw.githubusercontent.com/datasets/openml-datasets/ac945980fbb83bf4200a893593a6c1099679d150/data/diabetes/diabetes.csv>
2. **Glass Identification — multikelas.** Komposisi kimia kaca digunakan untuk membedakan enam jenis kaca.
   - Halaman dataset: <https://archive.ics.uci.edu/dataset/42/glass+identification>
   - CSV yang digunakan: <https://raw.githubusercontent.com/datasets/openml-datasets/ac945980fbb83bf4200a893593a6c1099679d150/data/glass/glass.csv>
3. **Yeast Protein Localization — multilabel.** Fitur protein digunakan untuk memprediksi 14 label lokasi/substrat sel. Satu protein dapat memiliki beberapa label.
   - Koleksi dataset: <https://github.com/tsoumakas/mulan/tree/43f6389095ad71984264db4387a8074b22bc7d4a/data/multi-label/yeast>
   - Berkas data: <https://github.com/tsoumakas/mulan/blob/43f6389095ad71984264db4387a8074b22bc7d4a/data/multi-label/yeast/yeast.arff>
