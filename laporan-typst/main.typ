// =====================================================================
// Template Laporan Tugas — Workshop Deep Learning (Teknik Telekomunikasi ITERA)
// Studi Ablasi RNN, LSTM, GRU vs ANN untuk Prediksi Trafik Seluler
//
// Format IEEE conference dua kolom (paket charged-ieee), MAKSIMAL 6 HALAMAN
// termasuk referensi.
// Compile: typst compile main.typ   (atau unggah folder ini ke typst.app)
// Bila muncul peringatan font "TeX Gyre Termes" saat compile lokal, pasang font TeX Gyre
// atau tambahkan --font-path ke folder font tersebut. Di typst.app font sudah tersedia.
//
// Teks abu-abu dalam #petunjuk[...] adalah petunjuk: HAPUS dan ganti dengan isi kalian.
// =====================================================================
#import "@preview/charged-ieee:0.1.4": ieee

#set text(lang: "id")
#let petunjuk(isi) = text(fill: gray.darken(20%))[[#isi]]

#show: ieee.with(
  title: [Studi Ablasi RNN, LSTM, dan GRU dibandingkan ANN untuk Prediksi Trafik Seluler Kota Milan],
  abstract: [
    Prakiraan trafik per sel membantu operator mengelola kapasitas jaringan. Studi ini membandingkan ANN, RNN, LSTM, dan GRU untuk meramal aktivitas internet per jam pada data Telecom Italia di kotak Duomo, Milan, melalui studi ablasi: satu hyperparameter diubah dalam satu waktu dari konfigurasi acuan. Lima faktor wajib (panjang jendela, ukuran _hidden_, jumlah lapisan, _learning rate_, kelompok fitur) dan dua faktor pilihan (horizon dan _gradient clipping_) diuji dengan tiga seed dan dilaporkan sebagai rata-rata $plus.minus$ simpangan baku. Konfigurasi terbaik dipilih dengan MAE validasi, lalu diuji pada minggu uji, minggu Natal, dan lokasi kelompok Corso Como. Pada Duomo keempat model mengalahkan baseline naif terbaik (test MAE 467,6). GRU terbaik dengan test MAE 220,3 ± 10,6, turun 7,1 % dari acuan, sedangkan ANN memperoleh perbaikan terbesar (12,5 %). Learning rate yang terlalu kecil, lapisan berlebih pada LSTM, dan jendela panjang pada ANN paling merugikan, sedangkan _hidden_ besar dan fitur waktu paling membantu. Menggabungkan nilai terbaik tidak selalu memperbaiki hasil. Di Corso Como urutan model berubah, dan semua model kalah dari baseline nilai terakhir pada minggu Natal.
  ],
  // Satu entri per anggota (maks. 3). Hapus yang tidak dipakai.
  authors: (
    (name: "Nizar Safariansyah", department: [NIM 124400029], organization: [Teknik Telekomunikasi, ITERA], email: "nizar.124400029@student.itera.ac.id"),
  ),
  index-terms: ("prediksi trafik seluler", "RNN", "LSTM", "GRU", "studi ablasi"),
  bibliography: bibliography("references.bib", title: [Pustaka]),
  figure-supplement: [Gambar],
)

= Pendahuluan
Prakiraan trafik per sel membantu operator mengalokasikan kapasitas dan merencanakan jaringan. Model sekuensial seperti RNN, LSTM @hochreiter1997lstm, dan GRU @cho2014gru memproses deret waktu jam demi jam, sedangkan ANN memperlakukan jendela masukan sebagai satu vektor datar. Membandingkan keempatnya dengan satu konfigurasi tidak menunjukkan seberapa jauh kesimpulan bergantung pada pilihan hyperparameter. Studi ablasi, yaitu mengubah satu faktor dalam satu waktu, menjawab pertanyaan itu. Kontribusi laporan ini: (1) ablasi tujuh faktor pada empat model dengan tiga seed; (2) pemilihan konfigurasi terbaik hanya dengan data validasi, lalu pengujian pada minggu uji dan Natal; dan (3) uji generalisasi pada lokasi lain di Milan.

= Data dan Metode

== Dataset
Data yang dipakai adalah _Telecom Italia Big Data Challenge_ @barlacchi2015milan dalam versi agregat per jam dari @zhang2021fedda. Setiap kotak grid Milan memiliki tiga deret waktu per jam: aktivitas internet (_net_), panggilan (_call_), dan SMS (_sms_), dengan periode 4 November sampai 29 Desember 2013. Variabel target adalah _net_. Bagian 0 sampai C memakai kotak Duomo (5060), sedangkan uji generalisasi memakai lokasi kelompok kode 9, Corso Como (kotak 5859), salah satu dari sepuluh lokasi tugas pada @fig:peta.

#figure(
  image("gambar/peta_lokasi_tugas.png", width: 85%),
  caption: [Rata-rata aktivitas internet per kotak di Milan. Lingkaran kuning: lokasi tugas (kode 0–9); lingkaran toska: Duomo.],
  placement: top,
) <fig:peta>

== Pembagian Data dan Praproses
Pembagian dilakukan berdasarkan waktu: data latih 11 November sampai 8 Desember (28 hari, 672 sampel), validasi 9 sampai 15 Desember, uji 16 sampai 22 Desember, dan minggu Natal 23 sampai 29 Desember. Minggu 4 sampai 10 November hanya menjadi konteks jendela masukan. Fitur dilengkapi jam dan hari dalam bentuk sin dan cos sehingga totalnya tujuh fitur, lalu distandarisasi dengan rata-rata dan simpangan baku dari data sebelum 9 Desember agar tidak ada informasi masa depan yang bocor. Sampel dibentuk dengan jendela geser: $W$ jam terakhir sebagai masukan dan nilai _net_ jam berikutnya (horizon 1) sebagai target. Data tidak diacak karena urutan waktu membawa informasi. Mengacak akan membuat model melihat masa depan saat berlatih, sehingga galat evaluasi menjadi terlalu optimistis, sedangkan pembagian kronologis meniru penggunaan nyata, yaitu meramal masa depan dari masa lalu.

== Model dan Konfigurasi Acuan
ANN meratakan jendela masukan menjadi satu vektor berukuran $W times 7$, lalu meneruskannya melalui lapisan tersembunyi berfungsi aktivasi ReLU menuju satu keluaran. Akibatnya jumlah parameter ANN bertambah seiring $W$. RNN, LSTM, dan GRU memproses jendela jam demi jam dengan bobot yang dipakai bersama pada setiap langkah, sehingga jumlah parameternya tidak bergantung pada $W$; keluaran diambil dari _hidden state_ jam terakhir lalu dipetakan oleh satu lapisan linear. LSTM dan GRU menambahkan gerbang pada RNN sederhana agar informasi jangka panjang lebih mudah dipertahankan @hochreiter1997lstm @cho2014gru. Ukuran _hidden_ acuan dipilih agar model rekuren memiliki sekitar 5,5 ribu parameter: ANN 6.497, RNN 5.305, LSTM 5.578, dan GRU 5.397. Semua model dilatih dengan fungsi loss MSE; _early stopping_ memantau MSE validasi dan memakai bobot terbaik. Konfigurasi acuan lengkap ada di @tab:acuan.

#figure(
  caption: [Konfigurasi acuan],
  placement: top,
  table(
    columns: 2,
    align: left,
    stroke: none,
    table.hline(),
    table.header([*Hyperparameter*], [*Nilai acuan*]),
    table.hline(stroke: 0.5pt),
    [Panjang jendela $W$], [24 jam],
    [Horizon], [1 jam],
    [Ukuran hidden], [ANN 32, RNN 68, LSTM 33, GRU 38],
    [Jumlah lapisan], [ANN 2, rekuren 1],
    [Fitur], [net, call, sms, jam, hari (7)],
    [Optimizer, _learning rate_], [Adam, $10^(-3)$],
    [_Batch size_], [64],
    [_Early stopping_], [_patience_ 30, maks. 300 epoch],
    table.hline(),
  ),
) <tab:acuan>

== Protokol Ablasi dan Evaluasi
Dari konfigurasi acuan pada @tab:acuan, hanya satu faktor diubah setiap kali. Faktor wajib (A1 sampai A5) adalah panjang jendela $W$ (6, 12, 24, 48, dan 168 jam), ukuran _hidden_ (16, 32, 64, dan 128), jumlah lapisan (1, 2, dan 3), _learning rate_ (0,0001, 0,001, dan 0,01), serta kelompok fitur (_net_; _net+call+sms_; _net+waktu_; dan semua fitur). Dua faktor pilihan adalah horizon (1, 6, dan 24 jam) dan _gradient clipping_ (tanpa, 0,5, dan 1,0) pada _learning rate_ 0,01. Hipotesis ditulis sebelum eksperimen dijalankan: _clipping_ 1,0 memperkecil simpangan baku antar seed dan sedikit memperbaiki val MAE model rekuren, sedangkan pada horizon 24 jam model hanya sedikit lebih baik atau bahkan kalah dari baseline "minggu lalu". Setiap konfigurasi dilatih dengan tiga seed (0, 1, dan 2) dan dilaporkan sebagai rata-rata $plus.minus$ simpangan baku. Metrik adalah MAE, RMSE, dan MAPE pada skala asli. Konfigurasi terbaik dipilih dengan val MAE rata-rata; minggu uji dan Natal hanya dipakai untuk pelaporan akhir. Seluruh pelatihan memakai CPU laptop tanpa GPU, dengan total waktu pelatihan sekitar 102 menit. Setiap run tercatat otomatis di `hasil/ablasi.csv` (369 baris; konfigurasi yang identik tidak dilatih ulang).

= Hasil

== Baseline Naif dan Acuan
Pada horizon 1 jam, tiga baseline naif di Duomo menghasilkan test MAE 600,9 (nilai terakhir), 552,2 (kemarin), dan 467,6 (minggu lalu); yang terbaik adalah tebakan minggu lalu. Konfigurasi acuan menghasilkan test MAE ANN $310 plus.minus 6$, RNN $230 plus.minus 5$, LSTM $274 plus.minus 25$, dan GRU $237 plus.minus 9$, sehingga keempat model mengalahkan baseline terbaik. Pada val MAE acuan, RNN ($206 plus.minus 7$) dan GRU ($208 plus.minus 4$) tidak dapat dibedakan karena selisihnya lebih kecil dari simpangan baku, sedangkan LSTM ($230 plus.minus 12$) dan ANN ($249 plus.minus 9$) lebih buruk.

== Ablasi Wajib
#figure(
  caption: [Val MAE (rata-rata $plus.minus$ simpangan baku, 3 seed) tiap faktor ablasi di Duomo.],
  placement: top,
  text(size: 7.5pt, table(
    columns: 6,
    align: (left, left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Faktor*], [*Nilai*], [*ANN*], [*RNN*], [*LSTM*], [*GRU*]),
    table.hline(stroke: 0.5pt),
    [$W$ (jam)], [6], [$213 plus.minus 5$], [$215 plus.minus 4$], [$227 plus.minus 4$], [$213 plus.minus 11$],
    [], [12], [$239 plus.minus 5$], [$207 plus.minus 6$], [$236 plus.minus 2$], [$209 plus.minus 10$],
    [], [24], [$249 plus.minus 9$], [$206 plus.minus 7$], [$230 plus.minus 12$], [$208 plus.minus 4$],
    [], [48], [$263 plus.minus 4$], [$200 plus.minus 5$], [$249 plus.minus 22$], [$207 plus.minus 5$],
    [], [168], [$298 plus.minus 18$], [$205 plus.minus 10$], [$241 plus.minus 8$], [$207 plus.minus 5$],
    [Hidden], [16], [$258 plus.minus 28$], [$247 plus.minus 6$], [$244 plus.minus 18$], [$217 plus.minus 21$],
    [], [32], [$249 plus.minus 9$], [$219 plus.minus 7$], [$241 plus.minus 21$], [$210 plus.minus 1$],
    [], [64], [$241 plus.minus 13$], [$207 plus.minus 13$], [$218 plus.minus 2$], [$198 plus.minus 9$],
    [], [128], [$234 plus.minus 1$], [$208 plus.minus 7$], [$207 plus.minus 8$], [$189 plus.minus 4$],
    [Lapisan], [1], [$241 plus.minus 8$], [$206 plus.minus 7$], [$230 plus.minus 12$], [$208 plus.minus 4$],
    [], [2], [$249 plus.minus 9$], [$207 plus.minus 9$], [$255 plus.minus 14$], [$203 plus.minus 10$],
    [], [3], [$249 plus.minus 12$], [$225 plus.minus 13$], [$339 plus.minus 17$], [$212 plus.minus 16$],
    [LR], [0,0001], [$295 plus.minus 5$], [$285 plus.minus 14$], [$399 plus.minus 25$], [$398 plus.minus 9$],
    [], [0,001], [$249 plus.minus 9$], [$206 plus.minus 7$], [$230 plus.minus 12$], [$208 plus.minus 4$],
    [], [0,01], [$262 plus.minus 17$], [$224 plus.minus 10$], [$216 plus.minus 14$], [$208 plus.minus 6$],
    [Fitur], [net], [$261 plus.minus 6$], [$245 plus.minus 6$], [$266 plus.minus 6$], [$265 plus.minus 7$],
    [], [net+call+sms], [$260 plus.minus 5$], [$224 plus.minus 2$], [$244 plus.minus 7$], [$232 plus.minus 13$],
    [], [net+waktu], [$246 plus.minus 9$], [$223 plus.minus 10$], [$231 plus.minus 4$], [$213 plus.minus 18$],
    [], [semua], [$249 plus.minus 9$], [$206 plus.minus 7$], [$230 plus.minus 12$], [$208 plus.minus 4$],
    [Horizon (jam)], [1], [$249 plus.minus 9$], [$206 plus.minus 7$], [$230 plus.minus 12$], [$208 plus.minus 4$],
    [], [6], [$337 plus.minus 2$], [$417 plus.minus 11$], [$393 plus.minus 24$], [$381 plus.minus 21$],
    [], [24], [$449 plus.minus 8$], [$437 plus.minus 3$], [$457 plus.minus 20$], [$452 plus.minus 12$],
    [Clip (LR 0,01)], [tanpa], [$262 plus.minus 17$], [$224 plus.minus 10$], [$216 plus.minus 14$], [$208 plus.minus 6$],
    [], [0,5], [$257 plus.minus 7$], [$229 plus.minus 9$], [$213 plus.minus 4$], [$218 plus.minus 17$],
    [], [1,0], [$247 plus.minus 10$], [$232 plus.minus 3$], [$224 plus.minus 21$], [$211 plus.minus 10$],
    table.hline(),
  )),
) <tab:ablasi>

@tab:ablasi merangkum val MAE seluruh faktor wajib. (A1) Memperpanjang jendela merugikan ANN, dari $213 plus.minus 5$ pada $W=6$ menjadi $298 plus.minus 18$ pada $W=168$, kemungkinan karena jumlah parameternya ikut membengkak sementara data latih hanya 672 sampel. RNN dan GRU hampir tidak terpengaruh (RNN 200 sampai 215, GRU 207 sampai 213), sedangkan LSTM naik turun dengan simpangan baku besar. (A2) Ukuran _hidden_ yang lebih besar umumnya membantu: pada _hidden_ 128, GRU membaik dari $217 plus.minus 21$ menjadi $189 plus.minus 4$ dan LSTM dari $244 plus.minus 18$ menjadi $207 plus.minus 8$, sedangkan RNN berhenti membaik di 64. Pada faktor ini jumlah parameter antar model tidak lagi setara. (A3) Menambah lapisan umumnya tidak menolong; LSTM paling terpengaruh, memburuk dari $230 plus.minus 12$ (1 lapisan) menjadi $339 plus.minus 17$ (3 lapisan). (A4) _Learning rate_ 0,0001 terburuk untuk semua model (LSTM $399 plus.minus 25$, GRU $398 plus.minus 9$) karena seluruh run berhenti di batas 300 epoch sebelum konvergen, sedangkan _learning rate_ 0,01 hanya menguntungkan LSTM ($216 plus.minus 14$). (A5) Memakai _net_ saja paling buruk di keempat model. Penambahan fitur waktu memberi perbaikan terbesar pada ANN, LSTM, dan GRU, sedangkan penambahan _call_ dan _sms_ saja hampir tidak membantu ANN.

== Ablasi Pilihan
*Horizon.* Hipotesis awal: MAE naik seiring horizon, dan pada 24 jam model hanya sedikit lebih baik atau bahkan kalah dari baseline minggu lalu. Hasil mendukung hipotesis: val MAE naik dari 206 sampai 249 pada horizon 1 menjadi 437 sampai 457 pada horizon 24, dan test MAE pada horizon 24 (ANN 527, RNN 486, LSTM 496, GRU 503) lebih buruk daripada baseline minggu lalu (466,1). Pada horizon 6 jam ANN justru terbaik ($337 plus.minus 2$) dibanding model rekuren (381 sampai 417), sehingga keunggulan model rekuren hanya tampak pada horizon 1 jam.

*Gradient clipping.* Hipotesis awal: _clipping_ 1,0 memperkecil simpangan baku dan sedikit memperbaiki val MAE model rekuren pada _learning rate_ 0,01. Hipotesis tidak terbukti. Dibanding tanpa _clipping_, val MAE RNN naik dari $224 plus.minus 10$ menjadi $232 plus.minus 3$, LSTM dari $216 plus.minus 14$ menjadi $224 plus.minus 21$, dan GRU dari $208 plus.minus 6$ menjadi $211 plus.minus 10$; simpangan baku hanya mengecil pada RNN. Perbaikan justru terjadi pada ANN ($262 plus.minus 17$ menjadi $247 plus.minus 10$), padahal ANN diperkirakan tidak terpengaruh. Seluruh selisih ini sebanding dengan simpangan baku antar seed, sehingga efek _clipping_ pada pengaturan ini tidak dapat dibedakan dari derau.

== Konfigurasi Terbaik
Konfigurasi terbaik dipilih dengan val MAE rata-rata dari seluruh hasil A1 sampai B7 (@tab:terbaik). Kombinasi nilai terbaik dari beberapa faktor dicoba lebih dulu, tetapi tidak selalu lebih baik: untuk GRU kombinasi ($W=48$, _hidden_ 128, dua lapisan) menghasilkan val MAE 213,8, lebih buruk daripada acuan (207,5), dan untuk ANN dan LSTM kombinasinya (232,6 dan 228,0) lebih buruk daripada satu faktor tunggal terbaik (212,5 dan 207,1). Konfigurasi akhir karena itu memakai satu faktor: ANN $W=6$, RNN $W=48$, serta LSTM dan GRU _hidden_ 128. Seluruh percobaan tercatat di `hasil/ablasi.csv` (bagian C dan C-coba1). Dibanding acuan, test MAE membaik pada keempat model: ANN 309,9 menjadi 271,1; RNN 229,6 menjadi 224,5 (selisih lebih kecil daripada simpangan baku); LSTM 273,6 menjadi 245,6; dan GRU 237,2 menjadi 220,3. Pada minggu Natal MAE naik sekitar dua kali lipat untuk semua model; baseline nilai terakhir (431,2) hanya dikalahkan oleh RNN (404,5), sedangkan baseline minggu lalu (1426,2) dikalahkan semuanya.

#figure(
  caption: [Acuan dan konfigurasi terbaik di Duomo (rata-rata $plus.minus$ simpangan baku, 3 seed).],
  placement: top,
  text(size: 7.5pt, table(
    columns: 5,
    align: (left, left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Konfigurasi*], [*Val MAE*], [*Test MAE*], [*Natal MAE*]),
    table.hline(stroke: 0.5pt),
    [ANN], [acuan], [248,9 ± 8,9], [309,9 ± 5,9], [680,1 ± 52,0],
    [], [$W=6$], [212,5 ± 5,2], [271,1 ± 13,9], [531,4 ± 79,6],
    [RNN], [acuan], [206,1 ± 7,4], [229,6 ± 5,1], [421,2 ± 2,4],
    [], [$W=48$], [200,1 ± 5,3], [224,5 ± 11,7], [404,5 ± 9,9],
    [LSTM], [acuan], [230,5 ± 12,2], [273,6 ± 25,4], [586,0 ± 56,8],
    [], [_hidden_ 128], [207,1 ± 8,4], [245,6 ± 6,7], [491,1 ± 7,6],
    [GRU], [acuan], [207,5 ± 3,8], [237,2 ± 9,3], [468,2 ± 21,8],
    [], [_hidden_ 128], [188,7 ± 4,2], [220,3 ± 10,6], [435,2 ± 7,5],
    table.hline(),
  )),
) <tab:terbaik>

== Uji Generalisasi
@tab:generalisasi memuat hasil di Corso Como memakai konfigurasi acuan dan konfigurasi terbaik Duomo tanpa diubah. Baseline naif di lokasi ini (test MAE) adalah 168,4 (nilai terakhir), 220,7 (kemarin), dan 233,8 (minggu lalu); keempat model mengalahkan ketiganya (test MAE 124,1 sampai 141,3). Urutan model berbeda dari Duomo: pada konfigurasi terbaik RNN terbaik di val MAE (136,3), disusul GRU (140,9), ANN (144,6), dan LSTM (144,9), sedangkan di Duomo GRU terbaik dan ANN terburuk. Konfigurasi terbaik Duomo hanya memperbaiki val MAE secara jelas pada ANN (157,7 menjadi 144,6); pada RNN, LSTM, dan GRU selisihnya sebanding dengan simpangan baku, dan test MAE GRU justru sedikit memburuk (125,2 menjadi 127,1). Pada minggu Natal hasilnya tidak konsisten: ANN dan GRU membaik, RNN dan LSTM memburuk, dan seluruh model kalah dari baseline nilai terakhir (101,6). Grafik pola trafik menunjukkan level trafik Corso Como lebih rendah daripada Duomo dan turun sejak awal Desember, sehingga MAE antar lokasi tidak sebanding secara langsung.

#figure(
  caption: [Acuan dan konfigurasi terbaik di Corso Como (rata-rata $plus.minus$ simpangan baku, 3 seed).],
  placement: top,
  text(size: 7.5pt, table(
    columns: 5,
    align: (left, left, center, center, center),
    stroke: none,
    table.hline(),
    table.header([*Model*], [*Konfigurasi*], [*Val MAE*], [*Test MAE*], [*Natal MAE*]),
    table.hline(stroke: 0.5pt),
    [ANN], [acuan], [157,7 ± 13,2], [149,2 ± 6,9], [243,1 ± 26,7],
    [], [terbaik], [144,6 ± 3,8], [132,4 ± 11,6], [184,8 ± 30,5],
    [RNN], [acuan], [139,0 ± 5,0], [127,8 ± 1,5], [255,0 ± 77,1],
    [], [terbaik], [136,3 ± 3,4], [124,1 ± 5,1], [263,9 ± 20,4],
    [LSTM], [acuan], [152,2 ± 14,2], [142,6 ± 25,6], [403,6 ± 40,9],
    [], [terbaik], [144,9 ± 5,7], [141,3 ± 13,6], [435,6 ± 61,8],
    [GRU], [acuan], [141,7 ± 9,0], [125,2 ± 6,4], [273,9 ± 22,0],
    [], [terbaik], [140,9 ± 2,7], [127,1 ± 7,9], [214,2 ± 9,6],
    table.hline(),
  )),
) <tab:generalisasi>

= Pembahasan
Faktor yang paling berpengaruh bergantung pada model. Untuk ANN, panjang jendela paling menentukan: val MAE memburuk dari 213 pada $W=6$ menjadi 298 pada $W=168$, sejalan dengan membengkaknya jumlah parameter pada data latih yang hanya 672 sampel. Untuk LSTM dan GRU, ukuran _hidden_ dan fitur waktu paling membantu, sedangkan menambah lapisan merugikan LSTM (230 menjadi 339 pada tiga lapisan). _Learning rate_ 0,0001 merugikan semua model karena pelatihan berhenti di batas 300 epoch sebelum konvergen; ini lebih mencerminkan batas epoch daripada kelemahan nilai itu sendiri. _Gradient clipping_ tidak memberi efek yang dapat dibedakan dari derau. Perbedaan antar model tidak selalu lebih besar daripada perbedaan antar seed: RNN dan GRU pada acuan (206 ± 7 dan 208 ± 4) tidak dapat dibedakan, dan LSTM memiliki simpangan baku test hingga 25, sehingga peringkat dengan tiga seed harus dibaca hati-hati. Sebaliknya, selisih ANN terhadap model rekuren pada acuan jelas lebih besar daripada simpangan bakunya.

Menggabungkan nilai terbaik tiap faktor tidak menjamin hasil terbaik. Kombinasi pertama kami memberi val MAE GRU 213,8, lebih buruk daripada acuan (207,5), padahal _hidden_ 128 saja menghasilkan 188,7. Dugaan kami, faktor-faktor tersebut saling berinteraksi dan model yang terlalu besar (151.809 parameter) mengalami _overfitting_ pada data latih yang kecil; dugaan ini belum kami uji. Pada minggu Natal pola trafik berbeda dari minggu biasa dan model tidak pernah melihat hari libur saat berlatih. Di Corso Como urutan model berubah dan keunggulan konfigurasi terbaik Duomo hanya tampak jelas pada ANN; pergeseran level trafik sejak awal Desember mungkin menjelaskannya, tetapi juga belum diuji.

Keterbatasan studi ini: satu kota dan satu pembagian waktu, hanya 672 sampel latih, tiga seed per konfigurasi, tiap faktor diuji sendiri-sendiri sehingga interaksi antar faktor tidak tercakup, jumlah parameter antar model tidak setara pada beberapa faktor, dan pemilihan konfigurasi bertumpu pada satu minggu validasi.

= Kesimpulan
Pada Duomo, keempat model mengalahkan baseline naif terbaik (test MAE 467,6). Setelah ablasi, GRU terbaik dengan test MAE 220,3 ± 10,6 (turun 7,1 % dari acuan 237,2), diikuti RNN 224,5, LSTM 245,6, dan ANN 271,1; perbaikan terbesar terjadi pada ANN (12,5 %) dan LSTM (10,2 %), sedangkan RNN nyaris tidak berubah. Pengaruh hyperparameter bergantung pada model, dan gabungan nilai terbaik per faktor tidak selalu lebih baik daripada satu faktor saja. Di Corso Como urutan model berubah (RNN terbaik di val dan test), dan seluruh model kalah dari baseline nilai terakhir pada minggu Natal, sehingga hasil Duomo tidak otomatis berlaku untuk lokasi lain.

#heading(numbering: none)[Kontribusi Anggota]
Nizar Safariansyah (individu): pengaturan dan pelaksanaan seluruh eksperimen, analisis hasil, dan penulisan laporan.

#heading(numbering: none)[Pernyataan Penggunaan AI]
Claude (Anthropic) dipakai sebagai pemandu langkah pengerjaan (menjalankan notebook dan mengelola repositori), membantu menyusun kerangka laporan, merumuskan draf awal hipotesis Bagian B, dan menyusun draf teks laporan yang kemudian diperiksa penulis. Seluruh angka berasal dari run penulis sendiri dan tercatat di `hasil/ablasi.csv`.

#heading(numbering: none)[Kode dan Data]
Kode, notebook beserta output, dan seluruh hasil run tersedia di #link("https://github.com/nizar326/dlw-tugas-124400029").
