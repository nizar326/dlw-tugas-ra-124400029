# Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler Milan

Tugas Workshop Deep Learning — Teknik Telekomunikasi ITERA.

## Kelompok

| Peran | Nama | NIM |
|---|---|---|
| Ketua (individu) | Nizar Safariansyah | 124400029 |

- **Kelas:** RA
- **Lokasi kelompok:** 9 — Corso Como (digit terakhir NIM terkecil)
- **Laporan:** `laporan-typst/main.pdf`

## Cara Menjalankan

```bash
uv sync
uv run jupyter lab
```

Buka `tugas_ablasi.ipynb`, isi `LOKASI_KELOMPOK`, lalu jalankan semua sel. Semua run tercatat di `hasil/ablasi.csv`.

Perangkat yang kami pakai: laptop Axioo Pongo 725V2 (CPU, tanpa GPU), total waktu training: ±102 menit (369 run).

## Ringkasan Hasil

Di Duomo, konfigurasi terbaik tiap model (dipilih dengan val MAE) menghasilkan test MAE: ANN 271,1; RNN 224,5; LSTM 245,6; GRU 220,3. Semuanya mengalahkan baseline naif terbaik di test (minggu lalu, 467,6). Menggabungkan nilai terbaik dari beberapa faktor tidak selalu lebih baik daripada satu faktor saja. Di Corso Como, urutan model berubah dan konfigurasi terbaik Duomo hanya membantu sebagian.

## Struktur

```
tugas_ablasi.ipynb      notebook eksperimen (di-commit beserta output)
data/                   data trafik per jam (Duomo + 10 lokasi tugas)
hasil/                  CSV seluruh run, tabel, dan grafik
laporan-latex/ atau laporan-typst/   sumber laporan + PDF
```

## Data

Telecom Italia Big Data Challenge (Barlacchi dkk., 2015, *Scientific Data*, lisensi ODbL), versi per jam dari Zhang dkk. (INFOCOM 2021). Disiapkan oleh pengajar di repo [mctosima/dl-workshop](https://github.com/mctosima/dl-workshop).

## Penggunaan Alat AI

Claude (Anthropic) dipakai sebagai pemandu langkah pengerjaan dan bantuan penjelasan; rincian ada di laporan.