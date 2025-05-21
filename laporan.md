# Laporan Proyek Machine Learning - Siti Septiyah Agustin

## Domain Proyek

Pada proyek ini, kita akan membangun model machine learning untuk **memprediksi risiko kredit** nasabah. Dengan menggunakan informasi nasabah seperti usia, pendapatan, status rumah, dan lainnya, model ini akan membantu lembaga keuangan untuk menilai apakah nasabah berisiko gagal bayar atau tidak.

Masalah prediksi risiko kredit sangat penting dalam dunia keuangan, karena bisa mempengaruhi kebijakan pemberian pinjaman, mengurangi kerugian yang dihasilkan dari kredit macet, dan memitigasi risiko bisnis.

Sektor perbankan memiliki peran vital dalam menjaga kestabilan ekonomi nasional, terutama melalui kegiatan penyaluran dana kredit kepada masyarakat. Namun, pemberian kredit tidak lepas dari risiko yang dikenal sebagai risiko kredit, yaitu potensi kerugian akibat kegagalan atau ketidaksanggupan debitur dalam memenuhi kewajiban pembayaran pinjaman. Risiko ini dapat berdampak serius terhadap kesehatan keuangan bank dan stabilitas sistem keuangan secara keseluruhan.
Oleh karena itu, penting bagi bank untuk memiliki sistem yang mampu mengklasifikasikan risiko kredit, guna membedakan antara nasabah dengan potensi good loan (nasabah yang mampu membayar tepat waktu) dan bad loan (nasabah yang berisiko gagal bayar). Klasifikasi ini membantu bank dalam mengambil keputusan strategis, seperti menentukan persetujuan pinjaman, menetapkan suku bunga, atau merancang program mitigasi risiko.

Penggunaan teknologi seperti machine learning, khususnya algoritma Random Forest, terbukti memberikan hasil yang akurat dan efisien dalam proses klasifikasi risiko kredit. Algoritma ini dapat memproses berbagai jenis data—baik numerik maupun kategorikal—dan telah menunjukkan tingkat akurasi tinggi dalam penelitian terdahulu, seperti yang dicapai oleh Wasono (2022) dengan akurasi 98.16% dan Prasojo & Haryatmi (2021) sebesar 83% dalam kasus serupa.
Penerapan sistem klasifikasi risiko kredit tidak hanya meningkatkan efisiensi operasional bank, tetapi juga memperkuat manajemen risiko secara proaktif, mengurangi potensi kredit macet, serta menjaga integritas portofolio pinjaman bank di tengah persaingan industri keuangan yang semakin kompleks.

Referensi: 
[Analisa Prediksi Kelayakan Pemberian Kredit Pinjaman dengan Metode Random Forest](https://teknosi.fti.unand.ac.id/index.php/teknosi/article/view/1555) 
[Manajemen Risiko Kredit Bagi Bank Umum](https://r.search.yahoo.com/_ylt=Awrx_j2mBxhoNgIAut7LQwx.;_ylu=Y29sbwNzZzMEcG9zAzMEdnRpZAMEc2VjA3Ny/RV=2/RE=1747614886/RO=10/RU=https%3a%2f%2fprosiding.seminar-id.com%2findex.php%2fsainteks%2farticle%2fdownload%2f520%2f518/RK=2/RS=jwwci.5l.2Yhc28CfQT7Y_crzps-)

## Business Understanding

### Problem Statements
1. Bagaimana kita dapat memprediksi apakah nasabah layak untuk mendapatkan kredit akan gagal bayar atau tidak berdasarkan data yang tersedia?
2. Model prediksi mana yang paling efektif dalam mengidentifikasi nasabah dengan risiko tinggi?

### Goals
1. Membangun model yang memprediksi kelayakan nasabah untuk mendapatkan kredit berdasarkan data nasabah.
2. Menggunakan beberapa algoritma untuk menemukan model yang memiliki akurasi terbaik dalam memprediksi risiko kredit.

### Solution Statements
- Menggunakan 3 algoritma yaitu **Random Forest classifier**, **Logistic Regression**, dan **Gradient Boosting classifier** untuk membandingkan performa model dan memilih model terbaik berdasarkan metrik evaluasi yang terukur.
- untuk mengoptimalkan kinerja model, Hyperparameter tuning dilakukan pada model terbaik yaitu pada model Random Forest Classifier
- metrik yang digunakan untuk evaluasi model prediksi risiko kredit mencakup akurasi, yang mengukur sejauh mana prediksi model sesuai dengan data aktual; precision, yang menilai seberapa banyak prediksi positif yang benar; recall, yang mengukur sejauh mana model menangkap semua kasus positif yang seharusnya ada; dan F1-score, yang merupakan rata-rata harmonis antara precision dan recall, memberikan keseimbangan dalam situasi ketidakseimbangan kelas, dengan semua metrik ini dihitung menggunakan confusion matrix untuk mendapatkan nilai true positive, true negative, false positive, dan false negative.


## Data Understanding
Dataset ini memiliki urgensi tinggi dalam dunia perbankan dan keuangan karena membantu mengoptimalkan penilaian risiko kredit, meminimalkan kemungkinan kegagalan bayar, dan menjaga stabilitas finansial lembaga keuangan. [Kaggle] (https://www.kaggle.com/datasets/omarc4gk/credit-risk-dataset).

1. Dataset Berisi total 32.581 bari dan 12 fitur. 
Dataset yang digunakan berisi informasi mengenai nasabah dan pinjaman mereka. Beberapa fitur penting dalam dataset ini adalah:
- person_age: Usia orang yang mengajukan pinjaman.
- person_income: Pendapatan tahunan orang tersebut.
- person_home_ownership: Status kepemilikan rumah (misalnya, SEWA, MILIK, HIPOTEK).
- person_emp_length: Lama bekerja dalam tahun.
- loan_intent: Tujuan dari pinjaman yang diajukan (misalnya, PERSONAL, PENDIDIKAN, MEDIS).
- loan_grade: Kelas pinjaman berdasarkan tingkat risiko (misalnya, A, B, C, D).
- loan_amnt: Jumlah pinjaman yang diminta.
- loan_int_rate: Suku bunga untuk pinjaman.
- loan_status: Status pinjaman (1 untuk disetujui, 0 untuk ditolak).
- loan_percent_income: Persentase dari pendapatan orang tersebut yang digunakan untuk pinjaman.
- cb_person_default_on_file: Apakah orang tersebut memiliki catatan default (Y untuk Ya, N untuk Tidak).
- cb_person_cred_hist_length: Panjang sejarah kredit orang tersebut dalam tahun.

penjelasan lebih lanjut:
| **Column**                    | **Description**                           | **Data Type** | **Range Details**                                                |
|-------------------------------|-------------------------------------------|---------------|------------------------------------------------------------------|
| **person_age**                 | Usia peminjam.                           | int64         | Rentang: 18 - 100 tahun.                                         |
| **person_income**              | Pendapatan tahunan peminjam.             | int64         | Rentang: bervariasi, tergantung sektor dan lokasi.               |
| **person_home_ownership**      | Status kepemilikan rumah.                | object        | Nilai: 'OWN', 'MORTGAGE', 'RENT'.                                |
| **person_emp_length**          | Lama pengalaman kerja (tahun).           | float64       | Rentang: 0 - 50 tahun, bisa mengandung NaN.                      |
| **loan_intent**                | Tujuan peminjaman uang.                  | object        | Nilai: 'PERSONAL', 'EDUCATION', 'BUSINESS', dll.                 |
| **loan_grade**                 | Kelas/penilaian risiko pinjaman.         | object        | Nilai: 'A', 'B', 'C', 'D', 'E', 'F', 'G'.                        |
| **loan_amnt**                  | Jumlah uang yang dipinjam.               | int64         | Rentang: bervariasi, ribuan hingga ratusan ribu USD/             |
| **loan_int_rate**              | Tingkat bunga pinjaman.                  | float64       | Rentang: 5% - 30%, tergantung profil peminjam.                   |
| **loan_status**                | Status pinjaman (lunas/gagal bayar).     | int64         | Nilai: 0 = gagal bayar, 1 = lunas.                               |
| **loan_percent_income**        | Persentase pendapatan untuk pinjaman.    | float64       | Rentang: 0% - 100%, tergantung pinjaman dan pendapatan.          |
| **cb_person_default_on_file**  | Catatan gagal bayar sebelumnya.          | object        | Nilai: 'Y' (pernah gagal bayar), 'N' (tidak pernah).             |
| **cb_person_cred_hist_length** | Panjang riwayat kredit peminjam (bulan). | int64         | Rentang: 0 - 300 bulan, lama keterlibatan dalam aktivitas kredit.|

2. melakukan EDA dengan fungsi df.describe() untuk memberikan ringkasan statistik deskriptif dari data numerik untuk memahami distribusi dan karakteristik data sebelum dilakukan pemodelan atau transformasi.
3. Kondisi data missing value pada kolom 
    - person_emp_length: 895
    - loan_int_rate    : 3116
    oleh karena itu diperlukan penanganan data yang hilang.
4. mengkategorikan fitur numerik dan kategorikal
    - Kolom Numerikal: 'person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_status', 'loan_percent_income', 'cb_person_cred_hist_length'
    - Kolom Kategorikal: 'person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file'
5. Visualisasi histogram pada kolom numerik
6. mengecek apakah terdapat data outlier dengan boxplot, variabel yang terdapat outlier
    - person_age
    - person_income
    - person_emp_length
    - loan_amnt
    - loan_int_rate
    - loan_percent_income
    - cb_person_cred_hist_length
Data ini akan digunakan untuk membangun model yang dapat memprediksi status pinjaman berdasarkan data nasabah.
7. melihat hubungan antara pasangan variabel dengan pairplot
8. melihat korelasi antar fitur numerik dengan heatmap korelasi

- teknik visualisai (histogram, bloxplot, heatmap korelasi) dan eda(memeriksa data, type data, missing value,) dilakukan untuk memahami data pada dataset credit_risk_dataset

## Data Preparation

Proses data preparation dilakukan dengan beberapa tahapan:
1. menangani missing value pada kolom 
    - person_emp_length
    - loan_int_rate
2. Menghapus data duplikat untuk memastikan data unik.
3.  Mendeteksi Outlier untuk melihat apakah mereka akan mempengaruhi hasil analisis atau model yang dibangun.
4. Melakukan encoding pada kolom kategorikal dengan menggunakan **LabelEncoder**.
5. Menskalakan data numerik dengan **StandarScaler** agar semua fitur memiliki rentang yang sama.
6. Melakukan SMOTE untuk menyeimbangkan distribusi kelas target yaitu fitur loan_status dengan cara menghasilkan sampel sintetik dari kelas minoritas, sehingga model pembelajaran mesin dapat dilatih dengan data yang lebih seimbang. 
SimpleImputer dengan strategi 'median' menggantikan nilai yang hilang dengan nilai median dari kolom tersebut, memastikan data yang hilang tidak menyebabkan masalah dalam pelatihan model. Median dipilih karena lebih tahan terhadap nilai ekstrem (outlier) dibandingkan dengan rata-rata (mean). Dalam beberapa kasus, data bisa memiliki distribusi yang tidak normal, dan menggunakan median akan lebih mewakili nilai tengah yang lebih stabil dibandingkan menggunakan rata-rata yang dipengaruhi oleh outlier.

## Splitting
Data dibagi menjadi **80% untuk pelatihan** dan **20% untuk pengujian**. Pembagian ini memastikan model diuji pada data yang belum dilihat sebelumnya.

## Modeling
Model yang digunakan untuk klasifikasi adalah 
**Random Forest**, 
Kelebihan : Random Forest menggunakan banyak pohon keputusan (decision trees), yang membantu mengurangi overfitting yang sering terjadi pada pohon keputusan tunggal Dengan menggunakan banyak pohon keputusan, Dapat menunjukkan fitur mana yang paling berpengaruh terhadap hasil klasifikasi. Robust terhadap outlier dan data tidak seimbang (imbalanced).
Kekurangan : Random Forest bisa menjadi model yang besar dan rumit, membuatnya lebih sulit untuk diinterpretasikan dibandingkan dengan model yang lebih sederhana . Karena menggunakan banyak pohon keputusan, Random Forest memerlukan lebih banyak waktu untuk melatih dan melakukan prediksi, terutama pada dataset besar. Lebih boros memori dan komputasi dibanding model linear.
**Random Forest** dipilih karena kemampuannya untuk menangani data besar dan kompleks dengan baik. Model ini memberikan kestabilan dalam prediksi karena menggabungkan hasil dari berbagai pohon keputusan. Hal ini mengurangi risiko overfitting dan memberikan hasil yang lebih baik dalam banyak kasus. Parameter seperti n_estimators (jumlah pohon) dan max_depth (kedalaman maksimum pohon) memengaruhi kemampuan model dalam belajar dan menghindari overfitting. Secara umum, Random Forest memberikan akurasi yang sangat baik pada data yang kompleks dan tidak terstruktur.
**Random Forest** adalah model ensemble yang menggunakan banyak pohon keputusan (decision trees). Setiap pohon dilatih pada subset acak dari data, dan prediksi akhir dihasilkan dengan menggabungkan prediksi dari semua pohon. Proses ini mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal karena variasi antar pohon yang dilatih pada data yang berbeda.
**Random Forest** mengurangi overfitting dan meningkatkan keakuratan prediksi dengan menggabungkan banyak pohon keputusan yang dilatih pada subset data yang berbeda. Hal ini memungkinkan model untuk menggeneralisasi lebih baik pada data uji dan mengurangi variansi, menghasilkan error yang lebih rendah dibandingkan dengan pohon keputusan tunggal.

**Logistic Regression**,
kelebihan : Cepat dilatih bahkan pada dataset besar. Koefisien regresi dapat digunakan untuk memahami pengaruh tiap variabel terhadap peluang gagal bayar. Sangat sesuai untuk masalah seperti credit risk yang hanya memiliki dua kemungkinan (lulus/tidak lulus pinjaman). Tidak memerlukan tuning parameter kompleks.
kekurangan :  Tidak mampu menangkap pola kompleks atau interaksi non-linear antar fitur tanpa transformasi. 
**Logistic Regression** Logistic Regression cocok sebagai baseline model karena efisien, cepat, dan hasilnya mudah dimengerti oleh pihak bisnis/keuangan. Sebagai model awal, Logistic Regression memberikan interpretasi awal yang baik dalam memahami hubungan fitur dan probabilitas gagal bayar.
Logistic Regression dipilih karena kesederhanaannya, kemampuannya untuk menghasilkan probabilitas yang dapat diinterpretasikan, serta cocok untuk kasus di mana hubungan antara variabel independen dan dependen dapat dianggap linier.

**Gradient Boosting Classifier**. 
kelebihan : Gradient Boosting terkenal sangat baik dalam meminimalkan kesalahan model karena membangun model secara bertahap dan mengoreksi kesalahan sebelumnya, Sangat efektif untuk data yang hubungan antar fiturnya kompleks.  Tidak sensitif terhadap korelasi antar fitur karena menggunakan decision tree. apat menghindari overfitting jika dikonfigurasi dengan benar (misalnya, lewat pengaturan learning rate dan jumlah tree).
kekurangan :  Karena model dibangun secara bertahap, pelatihan membutuhkan waktu yang lebih banyak dibanding Logistic Regression atau Random Forest. Model lebih kompleks sehingga interpretasinya tidak sejelas Logistic Regression.Kinerja model sangat bergantung pada pemilihan parameter seperti learning_rate, n_estimators, dan max_dept karena:
  - learning_rate
    Parameter ini menentukan seberapa besar kontribusi setiap pohon baru terhadap model keseluruhan. Nilai yang terlalu besar bisa menyebabkan model belajar terlalu cepat dan overfitting, sedangkan nilai yang terlalu kecil menyebabkan proses belajar menjadi lambat dan underfitting (model tidak mampu menangkap pola dengan baik). Oleh karena itu, pemilihan learning_rate harus seimbang, biasanya dikombinasikan dengan jumlah pohon (n_estimators) yang lebih banyak.  
  - n_estimators
    Ini adalah jumlah pohon yang akan dibangun dalam proses boosting. Semakin banyak pohon, model akan semakin kompleks dan akurat — jika dikombinasikan dengan learning_rate yang kecil. Namun, terlalu banyak estimator bisa membuat proses pelatihan lama dan berisiko overfitting jika tidak dikendalikan dengan baik.
  - max_depth
    Parameter ini mengontrol kedalaman setiap pohon. Pohon yang terlalu dalam akan menangkap noise dari data (overfitting), sementara pohon yang terlalu dangkal bisa gagal menangkap pola penting (underfitting). Oleh karena itu, pemilihan max_depth sangat penting untuk menjaga kompleksitas model tetap optimal.

Kami memilih Gradient Boosting karena adalah model boosting yang membangun pohon keputusan secara iteratif, di mana setiap pohon baru berfokus untuk mengoreksi kesalahan yang dibuat oleh pohon sebelumnya. Model ini sangat efektif untuk meningkatkan akurasi, bahkan dengan sedikit parameter yang perlu disesuaikan.
Dalam boosting, model dibangun secara bertahap. Setiap pohon baru mengoreksi kesalahan (residual) dari pohon sebelumnya. Model ini sangat kuat dalam menangani data dengan banyak fitur dan hubungan kompleks antara fitur dan target. Boosting bekerja dengan cara menyesuaikan bobot pohon untuk lebih berfokus pada kesalahan yang lebih besar. Salah satu parameter penting dalam Gradient Boosting adalah learning rate, yang mengontrol berapa besar kontribusi setiap pohon terhadap prediksi akhir. Learning rate yang tinggi dapat menyebabkan overfitting, sementara yang terlalu rendah memperlambat konvergensi model. Oleh karena itu, penting untuk memilih nilai learning rate yang tepat untuk keseimbangan antara kecepatan pelatihan dan akurasi.

- Proses Improvement:  untuk mengoptimalkan kinerja model, Hyperparameter tuning dilakukan pada model terbaik yaitu pada model Random Forest Classifier dengan mencari parameter terbaik untuk model yang digunakan. Setelah proses tuning, model dievaluasi dengan metrik terukur.
Proses hyperparameter tuning yang dilakukan bertujuan untuk meningkatkan kinerja Random Forest Classifier dengan menyesuaikan beberapa hyperparameter penting untuk memaksimalkan hasil prediksi pada data uji. Beberapa parameter yang dituning antara lain jumlah pohon (n_estimators), kedalaman pohon (max_depth), jumlah sampel minimum untuk membagi simpul (min_samples_split), dan jumlah sampel minimum di simpul daun (min_samples_leaf). Penyesuaian ini dilakukan untuk menghindari overfitting dan meningkatkan generalisasi model. Setelah tuning, model dilatih menggunakan data yang telah diresampling untuk mengatasi ketidakseimbangan kelas dan mengurangi bias terhadap kelas mayoritas.

## Evaluation

Metrik yang digunakan untuk mengevaluasi model adalah **Akurasi**, **Precision**, **Recall**, **F1-Score**, dan **confusion matrix**. Model yang diuji adalah **Random Forest Regression**, **Logistic Regression**, dan **Gradient Boosting Regrssion**.
### Metrik Evaluasi Model

1. **Akurasi (Accuracy)**
   Mengukur sejauh mana prediksi model sesuai dengan data aktual.
   \[
   \text{Akurasi} = \frac{TP + TN}{TP + TN + FP + FN}
   \]
   - **TP** = True Positives (jumlah prediksi yang benar sebagai positif)
   - **TN** = True Negatives (jumlah prediksi yang benar sebagai negatif)
   - **FP** = False Positives (jumlah prediksi yang salah sebagai positif)
   - **FN** = False Negatives (jumlah prediksi yang salah sebagai negatif)

2. **Precision**
   Mengukur seberapa banyak prediksi positif yang benar.
   \[
   \text{Precision} = \frac{TP}{TP + FP}
   \]
   Precision penting saat biaya kesalahan dalam memprediksi positif lebih tinggi.

3. **Recall**
   Mengukur seberapa banyak prediksi positif yang berhasil menangkap semua yang seharusnya positif.
   \[
   \text{Recall} = \frac{TP}{TP + FN}
   \]
   Recall penting untuk memastikan kita tidak melewatkan nasabah yang berisiko tinggi.

4. **F1-Score**
   Merupakan rata-rata harmonis dari Precision dan Recall, memberikan keseimbangan antara keduanya.
   \[
   \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   \]
   F1-Score memberikan gambaran umum performa model dalam kasus ketidakseimbangan kelas.

semua metrik ini dihitung menggunakan **confusion matrix** untuk mendapatkan nilai true positive, true negative, false positive, dan false negative.
- Proses Improvement: Hyperparameter tuning menggunakan RandomizedSearchCV dapat meningkatkan kinerja model dengan mencari parameter terbaik untuk model yang digunakan. Setelah proses tuning, model dievaluasi dengan metrik terukur.

### Hasil Evaluasi

Berdasarkan hasil yang diberikan, berikut adalah evaluasi ketiga model klasifikasi berdasarkan akurasi data latih (train) dan data uji (test)

### Evaluation Results:

| **Model**                        | **Akurasi Train (%)** | **Akurasi Test (%)** |
|----------------------------------|-----------------------|----------------------|
| **Gradient Boosting Classifier** | 90.75                 | 90.22                |
| **Logistic Regression**          | 76.60                 | 76.88                |
| **Random Forest Classifier**     | 98.91                 | 94.53                |


## Hasil evaluasi

  - Random Forest Classifier memberikan akurasi tertinggi baik pada data latih (98.91%) maupun data uji (94.53%). Model ini menunjukkan kinerja yang sangat baik, baik dalam mempelajari pola dari data latih maupun dalam generalisasi pada data uji, yang mengindikasikan bahwa model ini memiliki kemampuan prediksi yang sangat baik dan stabil.
  - Gradient Boosting Classifier juga menunjukkan performa yang baik dengan akurasi 90.75% pada data latih dan 90.22% pada data uji. Meskipun tidak sebaik Random Forest, model ini masih memiliki akurasi yang cukup tinggi dan sangat dekat antara akurasi data latih dan data uji, menunjukkan kemampuan generalisasi yang baik.
  - Logistic Regression memiliki akurasi yang lebih rendah dibandingkan dua model lainnya, yaitu 76.60% pada data latih dan 76.88% pada data uji. Meskipun memiliki hasil yang stabil antara data latih dan uji, model ini kurang efektif dibandingkan model ensemble seperti Random Forest dan Gradient Boosting, terutama dalam menangani data yang lebih kompleks dan besar.

### Interpretasi Hasil:

  - Random Forest adalah model yang paling unggul, menunjukkan akurasi yang sangat tinggi baik pada data latih maupun uji, dengan selisih yang kecil antara keduanya. Ini menunjukkan bahwa Random Forest mampu menangani kompleksitas data dengan baik dan memiliki kemampuan generalisasi yang sangat baik. Model ini sangat cocok untuk dataset besar dan kompleks, seperti pada prediksi risiko kredit.
  - Gradient Boosting juga memberikan hasil yang sangat baik dengan akurasi tinggi pada data latih dan uji. Namun, meskipun hasilnya hampir sebanding dengan Random Forest, Gradient Boosting sedikit lebih lambat dalam pelatihan dan dapat memerlukan lebih banyak sumber daya komputasi.
  - Logistic Regression meskipun stabil, tidak memiliki kemampuan yang cukup untuk menangani hubungan yang lebih kompleks dalam data. Meskipun akurasi data uji sedikit lebih tinggi daripada data latih, hasil ini masih lebih rendah dibandingkan dengan kedua model lainnya.

**Random Forest** adalah pilihan terbaik untuk model klasifikasi ini karena memberikan hasil yang paling baik dalam hal akurasi dan kestabilan antara data latih dan uji.

## Hyperparameter Tuning
untuk mengoptimalkan kinerja model, Hyperparameter tuning dilakukan pada model terbaik yaitu pada model Random Forest Classifier
beberapa parameter untuk mengoptimalkan performa model, antara lain:
  - bootstrap=True: Menunjukkan bahwa pemilihan sampel untuk setiap pohon dilakukan dengan bootstrap sampling (pengambilan sampel dengan pengembalian).
  - ccp_alpha=0.0: Mengatur nilai pruning dengan pengurangan kompleksitas pohon.
  - criterion='entropy': Digunakan untuk memilih fungsi pemisahan terbaik dalam pohon keputusan (entropi untuk mengukur impurity).
  - max_depth=11: Menetapkan kedalaman maksimum pohon. Tuning ini dapat membantu mencegah model terlalu rumit dan mengurangi overfitting.
  - max_features='log2': Membatasi jumlah fitur yang dapat digunakan untuk membagi setiap simpul pohon, dengan memilih logaritma basis 2 dari jumlah total fitur.
  - min_samples_leaf=4: Menentukan jumlah minimum sampel pada setiap simpul daun, yang membantu mencegah overfitting.
  - min_samples_split=10: Menentukan jumlah sampel minimum yang diperlukan untuk membagi simpul lebih lanjut.
  - n_estimators=220: Menetapkan jumlah pohon dalam hutan.

## Hasil Evaluasi tuning
### Setelah Tuning:

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| **0**     | 0.81          | 0.91       | 0.86         | 5051        |
| **1**     | 0.90          | 0.78       | 0.84         | 5080        |

- **Accuracy**: 84.63%
- **Macro Avg**: 0.85 Precision, 0.85 Recall, 0.85 F1-Score
- **Weighted Avg**: 0.85 Precision, 0.85 Recall, 0.85 F1-Score

### Sebelum Tuning:

| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| **0**     | 0.91          | 0.98       | 0.95         | 5051        |
| **1**     | 0.98          | 0.91       | 0.94         | 5080        |

- **Accuracy**: 94.53%
- **Macro Avg**: 0.95 Precision, 0.95 Recall, 0.95 F1-Score
- **Weighted Avg**: 0.95 Precision, 0.95 Recall, 0.95 F1-Score

Setelah tuning, model diuji pada data uji dan dibandingkan dengan model asli.
  - Accuracy (Tuned): 0.8463
Akurasi model setelah tuning adalah 84.63%. Artinya, model yang telah dituning berhasil memprediksi dengan benar 84.63% dari total data uji.

  - Accuracy (Original): 0.9453
Akurasi model asli adalah 94.53%. Ini menunjukkan bahwa model asli (sebelum tuning) memberikan hasil yang lebih baik secara keseluruhan.

### Interpretasi Hasil
  - Akurasi: Model Sebelum Tuning memiliki akurasi lebih tinggi (94.53%) dibandingkan dengan model setelah tuning (84.63%). Ini menunjukkan bahwa model sebelum tuning memiliki kemampuan yang lebih baik dalam memprediksi baik pada data latih maupun data uji.
  - Precision dan Recall:Precision untuk kelas 0 dan 1 menurun setelah tuning, dengan kelas 0 turun dari 0.91 menjadi 0.81, dan kelas 1 turun dari 0.98 menjadi 0.90.
  - Recall untuk kelas 1 menurun signifikan dari 0.91 menjadi 0.78 setelah tuning, yang mengindikasikan bahwa model setelah tuning lebih sering melewatkan kasus yang seharusnya terklasifikasi sebagai kelas 1.
  - F1-Score: F1-Score juga menunjukkan penurunan setelah tuning, dengan kelas 0 turun dari 0.95 menjadi 0.86 dan kelas 1 turun dari 0.94 menjadi 0.84.

Penurunan ini menggambarkan adanya trade-off antara precision dan recall setelah tuning, yang mungkin bertujuan untuk menghindari overfitting dan meningkatkan generalisasi model pada data baru.

- Alasan Penurunan Kinerja Setelah Tuning:
  1. Overfitting pada Model Sebelum Tuning: Model sebelum tuning mungkin terlalu menyesuaikan diri dengan data latih, sehingga menghasilkan akurasi yang sangat tinggi. Namun, hal ini bisa menyebabkan overfitting, di mana model tidak dapat menggeneralisasi dengan baik pada data uji.
  2. Proses tuning sering kali mengorbankan akurasi tinggi pada data latih untuk mencapai model yang lebih seimbang dan dapat generalize lebih baik pada data uji yang tidak terlihat sebelumnya. Penurunan akurasi pada data uji setelah tuning menunjukkan bahwa model lebih stabil dan dapat diandalkan dalam prediksi yang lebih luas.

### Kesimpulan 
Dengan menggunakan Random Forest Classifier sebagai model terbaik, kami dapat memprediksi risiko kredit dengan akurasi yang lebih tinggi, yang akan membantu lembaga keuangan dalam membuat keputusan pemberian pinjaman yang lebih baik. Penggunaan teknik hyperparameter tuning dan evaluasi dengan berbagai metrik memberikan gambaran yang jelas mengenai kinerja model dan memastikan bahwa model yang dipilih dapat menggeneralisasi dengan baik pada data baru.
Secara keseluruhan, proses **hyperparameter tuning** pada **Random Forest Classifier** berhasil meningkatkan kestabilan dan kemampuan generalisasi model, meskipun ada sedikit penurunan akurasi pada data uji setelah tuning. Model sebelum tuning menunjukkan akurasi yang sangat tinggi, namun cenderung overfit pada data latih, sedangkan setelah tuning, meskipun akurasi menurun, model menjadi lebih seimbang dengan kemampuan yang lebih baik dalam menggeneralisasi data yang tidak terlihat sebelumnya. Meskipun ada trade-off dalam precision dan recall, tuning memungkinkan model untuk lebih stabil dan efektif dalam memprediksi risiko kredit di dunia nyata, mengurangi bias terhadap kelas mayoritas dan memperbaiki performa pada data yang lebih luas.



