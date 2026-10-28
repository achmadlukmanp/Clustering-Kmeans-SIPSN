# Dashboard Klasterisasi Pengelolaan Sampah (K-Means)

## Cara Menjalankan

1. (Opsional) Buat virtual environment:
   ```
   python -m venv env
   env\Scripts\activate   # Windows
   source env/bin/activate  # macOS/Linux
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi:
   ```
   python app.py
   ```
4. Buka browser ke `http://127.0.0.1:5000`

## Struktur Folder

```
project/
├── app.py
├── README.md
├── requirements.txt
├── static/
│   └── style.css
└── templates/
    ├── base.html
    ├── evaluation.html
    ├── home.html
    ├── interpretation.html
    ├── modeling.html
    ├── processing.html
    └── upload.html
```

## Ringkasan Perubahan (revisi terbaru)

- Upload Data: tambah catatan format file harus seragam; judul kartu "Berkas Data" → "Upload Data"; teks pemilihan variabel dipersingkat & lebih tegas.
- Processing Data: tambah penjelasan cek missing value (NaN tidak lanjut ke agregasi); missing value kini diimputasi dengan **nilai 0** (bukan mean); teks agregasi & standarisasi disederhanakan.
- Pemodelan: teks slider k disederhanakan; layout kartu Grafik Elbow & Silhouette diselaraskan.
- Evaluasi Model: judul panduan DBI dirapikan (hapus "(acuan umum)").
- Interpretasi Klaster: intro disederhanakan untuk pengguna awam; header tabel provinsi "K1/K2" → "Klaster 1/Klaster 2"; ditambahkan penjelasan singkat pada beberapa tabel.
