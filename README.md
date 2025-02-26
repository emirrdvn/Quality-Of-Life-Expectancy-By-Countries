# Yaşam Kalitesi Analizi ve Tahmini

Bu proje, ülkeler arasındaki yaşam kalitesi verilerini analiz etmeye ve modellemeye odaklanmıştır. Farklı sınıflandırma ve regresyon algoritmaları kullanılarak **Yaşam Kalitesi Değeri** ve **Yaşam Kalitesi Kategorisi** tahmin edilmektedir.

---

## Proje Amaçları

1. **Veri Analizi ve Görselleştirme**:
   - Eksik verilerin düzenlenmesi ve veri setinin temizlenmesi.
   - Özellikler arasındaki ilişkilerin korelasyon analizi.
   - Histogram, kutu grafiği (box-plot) ve ısı haritaları ile verinin görülebilir hale getirilmesi.

2. **Sınıflandırma Problemi**:
   - **Yaşam Kalitesi Kategorisi** tahmini için makine öğrenimi modelleri.
   - Kullanılan Modeller:
     - Lojistik Regresyon
     - Destek Vektör Sınıflandırıcı (SVC)
     - Rastgele Orman (Random Forest)
     - XGBoost

3. **Regresyon Problemi**:
   - **Yaşam Kalitesi Değeri** tahmini için regresyon modelleri.
   - Kullanılan Modeller:
     - Doğrusal Regresyon
     - Rastgele Orman Regresyonu
     - XGBoost Regresyonu

4. **Model Performans Değerlendirme**:
   - **Sınıflandırma Metrikleri**:
     - Doğruluk (Accuracy)
     - F1 Skoru
     - ROC AUC
   - **Regresyon Metrikleri**:
     - Ortalama Kare Hatası (MSE)
     - Kök Ortalama Kare Hatası (RMSE)
     - Ortalama Mutlak Hata (MAE)
     - R-kare (R²)

---

## Gereksinimler

Gerekli kütüphaneleri aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install pandas numpy scikit-learn xgboost seaborn matplotlib
```

---

## Dosya Yapısı

- **Quality_of_Life.csv**: Projede kullanılan veri seti.
- **QualityOfLifeSon.ipynb**: Analiz, modelleme ve değerlendirme için ana kod dosyası.

---

## Adım Adım İşlemler

### 1. Veri Setinin Yüklenmesi

Veri kümesi `pandas` ile yüklenir:
```python
file_path = "Quality_of_Life.csv"
df = pd.read_csv(file_path)
```

### 2. Veri Setinin İncelenmesi

Veri seti hakkında genel bilgiler ve eksik veri kontrolü yapılır:
```python
print(df.info())
print(df.isnull().sum())
```

![image](https://github.com/user-attachments/assets/ef23e1d4-9541-4225-86ef-5147003ebd07)

### 3. Eksik Değerlerin Doldurulması

- Sayısal sütunlardaki eksik değerler medyan ile doldurulur.
- Kategorik sütunlardaki eksik değerler "Unknown" ile doldurulur:
```python
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)
```

![image](https://github.com/user-attachments/assets/6c9c0e26-fb5d-4210-ac17-af2fca17491d)

### 4. Kategorik Verilerin Sayısallaştırılması

Kategorik veriler, `LabelEncoder` kullanılarak sayısal değerlere dönüştürülür:
```python
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
```

### 5. Hedef ve Özellik Değişkenlerinin Ayrılması

Tahmin yapılacak hedef değişken (örneğin `Quality of Life Category`) belirlenir ve veri seti bağımlı (X) ve bağımsız (y) değişkenler olarak ayrılır:
```python
target = "Quality of Life Category"
X = df.drop(columns=[target])
y = df[target]
```

### 6. Eğitim ve Test Setlerine Ayırma

Veri seti, eğitim ve test setleri olarak ikiye ayrılır:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 7. Verilerin Standartlaştırılması

Model performansını artırmak için veriler standartlaştırılır:
```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 8. Korelasyon Matrisi

Veri seti sütunlarının korelasyon matris gösterimi:
```python
corr_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Korelasyon Matrisi")
plt.show()
```

![image](https://github.com/user-attachments/assets/b6636ea2-7e45-41b2-ac83-2859d91c914d)


### 9. Modellerin Tanımlanması

Burada sınıflandırma algoritmaları Kullanılacaktır.
Veri seti üzerinde kullanılacak Logistic Regression,SVC,Random Forest ve XGBoost algoritmalarının tanımlanması:
```python
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Classifier": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}
```

### 10. Modelleri Eğitme

Kullanılacak modelleri döngü ile eğitip değerlendirm:
```python
for model_name, model in models.items():
    print(f"\nEğitiliyor: {model_name}")

    # Model eğitimi
    model.fit(X_train, y_train)

    # Tahminler
    y_pred = model.predict(X_test)

    # Olasılık veya karar fonksiyonu kontrolü
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
    elif hasattr(model, "decision_function") and y_test.nunique() == 2:  # İkili sınıflandırma kontrolü
        y_proba = model.decision_function(X_test)
        auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        auc = np.nan  # ROC AUC hesaplanamaz

    # Performans metrikleri
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Sonuçları kaydetme
    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    # Detaylı sınıflandırma raporu
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
```

![image](https://github.com/user-attachments/assets/c7bd5cb5-9639-43c2-96b7-78d2f741b4d3)


### 11.Sonuçları Grafik Olarak Gösterme ve Karşılaştırma

Elde ettiğimiz modellerin sonuçlarını grafik haline dönüştürme ve kıyaslama:
```python
results_df = pd.DataFrame(results)
print("\nModel Karşılaştırma Sonuçları:")
print(results_df)
results_df.set_index("Model", inplace=True)
results_df.plot(kind="bar", figsize=(12, 8), alpha=0.8)
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Değer")
plt.xlabel("Modeller")
plt.xticks(rotation=45)
plt.legend(title="Metrikler")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/e7d483fa-4683-4107-b5df-fb1694aa23a0)


### 12. Model Eğitmeyi Quality Of Life Value Üzerinden Regresyon ile Gerçekleştirme

Bu sefer target olarak Quality Of Life Catefory yerine Quality Of Life Value seçiyoruz ve devam ediyoruz.
Yine yukarıda yaptığımız işlemlerden bazılarını tekrar ediyoruz:
```python
X = df.drop(columns=["Quality of Life Value"])  # Diğer tüm özellikler
y = df["Quality of Life Value"]  # Hedef değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 13. Regresyon Modellerini Tanımlama

Gerekli regresyon modellerini seçiyoruz:
```python
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}
```

### 14. Regresyon Modellerinin Eğitilmesi

Regresyon modellerini eğitiyoruz:
```python
for model_name, model in models.items():
    # Model eğitimi
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları kaydetme
    results.append({
        "Model": model_name,
        "Mean Squared Error (MSE)": mse,
        "R-squared (R2)": r2
    })
```

![image](https://github.com/user-attachments/assets/dd9c7470-1054-4bd9-8170-620026db226b)


### 15. Mean Squared Error ve R-Squared Sonuçları,Grafikleri ve Karşılaştırılması 

Elde ettiğimiz lineer regresyon, Random Forest ve XBG regresyon modellerinin sonuçları:
```python
results_df = pd.DataFrame(results)
print("\nModel Performans Karşılaştırması:")
print(results_df)
results_df2= results_df.drop(columns=["Mean Squared Error (MSE)"])
results_df3 = results_df.drop(columns=["R-squared (R2)"])
results_df3.set_index("Model", inplace=True)
results_df3.plot(kind="bar", figsize=(12, 6), alpha=0.8)
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Değer")
plt.xticks(rotation=45)
plt.legend(title="Metrikler")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
results_df2.set_index("Model", inplace=True)
results_df2.plot(kind="bar", figsize=(12, 6), alpha=0.8)
plt.title("Model Performans Karşılaştırması")
plt.ylabel("Değer")
plt.xticks(rotation=45)
plt.legend(title="Metrikler")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/09ecd9a7-b29d-4a0f-bf73-a2aa624759b8)
![image](https://github.com/user-attachments/assets/3740a78f-8872-499a-a4a8-59c92fc7864c)
<br/>
![image](https://github.com/user-attachments/assets/95483979-1926-456f-9d92-a5ce46e37a38)

# Sonuç Analizi

## **QUALITY OF LIFE CATEGORY**

### **1. Lojistik Regresyon**
- **Doğruluk (Accuracy)**: %81.25
- **F1-Skoru**: 0.80
- **Ana Noktalar**:
  - Dengeli bir model; sınıf 4 ve 5 için güçlü sonuçlar vermektedir.
  - Sınıf 0 (%25) ve sınıf 1 (%38) için düşük bir **recall** değerine sahiptir.
  - Basit veri setleri için uygun, hızlı bir yöntemdir.

### **2. Destek Vektör Makinesi (SVC)**
- **Doğruluk (Accuracy)**: %75.00
- **F1-Skoru**: 0.70
- **Ana Noktalar**:
  - Sınıf 3, 4 ve 5 için güçlü bir performans göstermektedir.
  - Doğrusal olmayan karar sınırları için faydalı olabilir ancak bu veri setinde sınırlı bir başarı göstermiştir.

### **3. Rastgele Orman (Random Forest)**
- **Doğruluk (Accuracy)**: %87.50
- **F1-Skoru**: 0.87
- **Ana Noktalar**:
  - Çoğu sınıfta dengeli bir performans göstermektedir.
  - Çok yönlü ve genelleştirilebilir bir modeldir; metrikler arasında tutarlı sonuçlar vermektedir.

### **4. XGBoost**
- **Doğruluk (Accuracy)**: %97.92
- **F1-Skoru**: 0.98
- **Ana Noktalar**:
  - Tüm sınıflarda yüksek precision ve recall değerleri ile en iyi performansı sergilemektedir.
  - Hesaplama açısından daha maliyetli ancak üstün sonuçlar sağlamaktadır.
  - Bu veri seti için en uygun modeldir.

## **Sonuç**

- **En İyi Model**: XGBoost, doğruluk (%97.92) ve F1-skoru (0.98) açısından diğer modelleri geride bırakarak en uygun model olarak öne çıkmaktadır.
- **En İyi İkinci Model**: Derste işlediğimiz modellerden Random Forest Sınıflandırma algoritması doğruluk (%87.50) ve F1-skoru (0.87) ile iyi bir precision vermiştir.

## **QUALITY OF LIFE VALUE**

### **1. Doğrusal Regresyon (Linear Regression)**
- **Mean Squared Error (MSE)**: 277.1227
- **Root Mean Squared Error (RMSE)**: 16.6470
- **Mean Absolute Error (MAE)**: 12.5057
- **R-squared (R²)**: 0.8169

**Ana Noktalar:**
- Basit ve hızlı bir modeldir.
- Diğer modellere kıyasla daha yüksek hata oranına sahiptir.
- Veri setindeki karmaşık ilişkileri yeterince yakalayamamıştır.

### **2. Rastgele Orman (Random Forest)**
- **Mean Squared Error (MSE)**: 59.8929
- **Root Mean Squared Error (RMSE)**: 7.7390
- **Mean Absolute Error (MAE)**: 4.5648
- **R-squared (R²)**: 0.9604

**Ana Noktalar:**
- Çok yönlü bir modeldir ve genelleştirme kapasitesi yüksektir.
- En düşük hata oranına sahiptir.
- Veri setindeki karmaşık ilişkileri başarıyla öğrenmiştir.

### **3. XGBoost**
- **Mean Squared Error (MSE)**: 112.0884
- **Root Mean Squared Error (RMSE)**: 10.5872
- **Mean Absolute Error (MAE)**: 5.1482
- **R-squared (R²)**: 0.9259

**Ana Noktalar:**
- Hata oranı Rastgele Orman modeline kıyasla biraz daha yüksektir.
- Daha hesaplamalı bir modeldir ancak güçlü sonuçlar verir.
- Büyük ve karmaşık veri setlerinde avantaj sağlar.

---

## **Performans Tablosu**

| Model               | MSE      | RMSE     | MAE      | R²       |
|---------------------|----------|----------|----------|----------|
| Linear Regression   | 277.1227 | 16.6470  | 12.5057  | 0.8169   |
| Random Forest       | 59.8929  | 7.7390   | 4.5648   | 0.9604   |
| XGBoost             | 112.0884 | 10.5872  | 5.1482   | 0.9259   |

---

## **Sonuç**
- **En İyi Model**: Rastgele Orman, en düşük hata oranları ve en yüksek R² değeri ile en iyi performansı sergilemiştir.
- **XGBoost**, doğruluk açısından ikinci sırada yer almakta ve büyük veri setleri için güçlü bir alternatif sunmaktadır.
- **Doğrusal Regresyon**, basit ve hızlı bir çözüm sunmakla birlikte, daha karmaşık modellerin gerisinde kalmıştır.

## XGBoost Nedir ?
XGBoost nedir?
XGBoost (eXtreme Gradient Boosting), gradyan inişinden faydalanan denetlenen bir öğrenme artırma algoritması olan gradyan artırılmış karar ağaçlarını kullanan dağıtılmış, açık kaynaklı bir makine öğrenme kütüphanesidir. Hızı, verimliliği ve büyük veri kümeleriyle iyi ölçeklenebilme yeteneğiyle bilinir.

Washington Üniversitesi'nden Tianqi Chen tarafından geliştirilen XGBoost, aynı genel çerçeveye sahip gelişmiş bir gradyan güçlendirme uygulamasıdır; yani, zayıf öğrenen ağaçlarını kalıntıları ekleyerek güçlü öğrenenlerle birleştirir. Kütüphane C++, Python, R, Java, Scala ve Julia 1 için mevcuttur .

Karar ağaçları ve güçlendirme
Karar ağaçları, makine öğreniminde sınıflandırma veya regresyon görevleri için kullanılır. Bir iç düğümün bir özelliği, dalın bir karar kuralını ve her yaprak düğümün veri kümesinin sonucunu temsil ettiği hiyerarşik bir ağaç yapısı kullanırlar.

Karar ağaçları aşırı uyuma eğilimli olduğundan , güçlendirme gibi topluluk yöntemleri genellikle daha sağlam modeller oluşturmak için kullanılabilir. Güçlendirme, birden fazla zayıf ağacı birleştirir; yani, rastgele şanstan biraz daha iyi performans gösteren modeller, güçlü bir öğrenen oluşturur. Her zayıf öğrenen, önceki modellerin yaptığı hataları düzeltmek için sırayla eğitilir. Yüzlerce yinelemeden sonra, zayıf öğrenenler güçlü öğrenenlere dönüştürülür.

Rastgele ormanlar ve güçlendirme algoritmaları, tahmin performansını iyileştirmek için bireysel öğrenen ağaçları kullanan popüler topluluk öğrenme teknikleridir. Rastgele ormanlar, torbalama (önyükleme toplama) kavramına dayanır ve tahminlerini birleştirmek için her ağacı bağımsız olarak eğitirken, güçlendirme algoritmaları, zayıf öğrenenlerin önceki modellerin hatalarını düzeltmek için sıralı olarak eğitildiği bir katkısal yaklaşım kullanır.
<br/>
[IBM XGBoost Sayfası](https://www-ibm-com.translate.goog/think/topics/xgboost?_x_tr_sl=en&_x_tr_tl=tr&_x_tr_hl=tr&_x_tr_pto=tc)
<br/>
![image](https://github.com/user-attachments/assets/3aa7a380-59b9-4321-b646-59acfd439027)

## Genel Çalışma Adımları

1. **Veri Yükleme ve Ön işleme**:
   - `Quality_of_Life.csv` dosyasını yükleyin.
   - Eksik değerleri doldurun.
   - Kategorik değişkenleri `LabelEncoder` ile sayısallaştırın.
   - Veri setini bağımlı ve bağımsız değişkenlere ayırın.

2. **Görselleştirme**:
   - Korelasyon matrisi ısı haritası.
   - Veri dağılımı için histogramlar ve kutu grafikleri.

3. **Sınıflandırma Modelleri**:
   - Sınıflandırma görevi için eğitim ve test veri seti oluşturun.
   - Modelleri eğitin: Lojistik Regresyon, SVC, Rastgele Orman, XGBoost.
   - Modelleri Doğruluk, F1 Skoru ve ROC AUC ile değerlendirin.

4. **Regresyon Modelleri**:
   - Regresyon görevi için eğitim ve test veri seti oluşturun.
   - Modelleri eğitin: Doğrusal Regresyon, Rastgele Orman Regresyonu, XGBoost Regresyonu.
   - Modelleri MSE, RMSE, MAE ve R² ile değerlendirin.

5. **Model Karşılaştırması**:
   - Performans metriklerini bar grafiklerle görün.

---

## Kullanılan Modeller

### Sınıflandırma Algoritmaları
- Lojistik Regresyon
- Destek Vektör Sınıflandırıcı (SVC)
- Rastgele Orman Sınıflandırıcı
- XGBoost Sınıflandırıcı

### Regresyon Algoritmaları
- Doğrusal Regresyon
- Rastgele Orman Regresyonu
- XGBoost Regresyonu

---

## Görselleştirme

Performans metrikleri, modellerin kolay karşılaştırılması için bar grafikleri şeklinde gösterilir:

- **Sınıflandırma Metrikleri**:
  - Doğruluk
  - F1 Skoru
  - ROC AUC

- **Regresyon Metrikleri**:
  - Ortalama Kare Hatası (MSE)
  - Kök Ortalama Kare Hatası (RMSE)
  - Ortalama Mutlak Hata (MAE)
  - R-kare (R²)

---

## Nasıl Çalıştırılır

1. **Veri Setini Yerleştirin**:
   `Quality_of_Life.csv` dosyasını çalışma dizinine kopyalayın.

2. **Kodu Çalıştırın**:
   Notebook'u Jupyter Notebook, Google Colab veya herhangi bir Python IDE'de çalıştırın.

3. **Sonuçları Analiz Edin**:
   Çıktı metriklerini ve grafiklerini gözlemleyerek içgörüler elde edin.

---

## Yazarlar ve Katkılar

Bu proje, yaşam kalitesi metriklerini analiz etmek ve tahmin etmek amacıyla geliştirilmiştir. Geri bildirimler ve katkılar memnuniyetle karşılanır!
