# PageSpeed Insights Analyzer

PageSpeed Insights Analyzer adalah aplikasi berbasis Streamlit untuk menganalisa performa website menggunakan Google PageSpeed Insights API. Anda dapat mengunggah sitemap dan melakukan analisa massal pada banyak halaman secara sekaligus. Hasil analisa dapat diunduh dalam bentuk file Excel.

## Fitur
- Analisa performa website menggunakan Google PageSpeed Insights API (mobile & desktop)
- Upload sitemap (XML) untuk analisa banyak URL sekaligus
- Progres analisa real-time dan dapat dijalankan di background
- Hasil analisa detail dan ringkasan dalam format Excel (.xlsx)
- Tampilan antarmuka interaktif berbasis Streamlit

## Cara Install

1. Kloning repo ini:
   ```bash
   git clone https://github.com/mrohadiz/pagespeed-insights-analyzer.git
   cd pagespeed-insights-analyzer
   ```

2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Jalankan aplikasi:
   ```bash
   streamlit run main.py
   ```

## Cara Menggunakan

1. Siapkan API Key Google PageSpeed Insights (lihat [dokumentasi resmi](https://developers.google.com/speed/docs/insights/v5/get-started)).
2. Masukkan URL sitemap website Anda.
3. Atur jumlah URL yang ingin dianalisa (opsional).
4. Pilih mode analisa (mobile/desktop).
5. Klik "Start Analysis".
6. Download laporan hasil analisa setelah selesai.

## Struktur Project

- `main.py` - Entry point aplikasi Streamlit
- `api.py` - Modul untuk koneksi ke PageSpeed Insights API
- `data_processing.py` - Pengolahan dan rekap data hasil analisa
- `utils.py` - Fungsi utilitas, logging, validasi, dsb.
- `ui.py` - Komponen UI Streamlit
- `requirements.txt` - Daftar dependensi Python

## Lisensi

MIT License

---

**Dibuat oleh [mrohadiz](https://github.com/mrohadiz)**
