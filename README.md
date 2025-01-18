# Yaşam Kalitesi Analizi ve Tahmini

Bu proje, ülkeler arasındaki yaşam kalitesi verilerini analiz etmeye ve modellemeye odaklanmıştır. Farklı sınıflandırma ve regresyon algoritmaları kullanılarak **Yaşam Kalitesi Değeri** ve **Yaşam Kalitesi Kategorisi** tahmin edilmektedir.

---
## YouTube Tanıtım Videosu

[Proje Tanıtım Videosu](linkburaya)

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

## Sonuç Analizi

## XGBoost Nedir ?
XGBoost nedir?
XGBoost (eXtreme Gradient Boosting), gradyan inişinden faydalanan denetlenen bir öğrenme artırma algoritması olan gradyan artırılmış karar ağaçlarını kullanan dağıtılmış, açık kaynaklı bir makine öğrenme kütüphanesidir. Hızı, verimliliği ve büyük veri kümeleriyle iyi ölçeklenebilme yeteneğiyle bilinir.

Washington Üniversitesi'nden Tianqi Chen tarafından geliştirilen XGBoost, aynı genel çerçeveye sahip gelişmiş bir gradyan güçlendirme uygulamasıdır; yani, zayıf öğrenen ağaçlarını kalıntıları ekleyerek güçlü öğrenenlerle birleştirir. Kütüphane C++, Python, R, Java, Scala ve Julia 1 için mevcuttur .

Karar ağaçları ve güçlendirme
Karar ağaçları, makine öğreniminde sınıflandırma veya regresyon görevleri için kullanılır. Bir iç düğümün bir özelliği, dalın bir karar kuralını ve her yaprak düğümün veri kümesinin sonucunu temsil ettiği hiyerarşik bir ağaç yapısı kullanırlar.

Karar ağaçları aşırı uyuma eğilimli olduğundan , güçlendirme gibi topluluk yöntemleri genellikle daha sağlam modeller oluşturmak için kullanılabilir. Güçlendirme, birden fazla zayıf ağacı birleştirir; yani, rastgele şanstan biraz daha iyi performans gösteren modeller, güçlü bir öğrenen oluşturur. Her zayıf öğrenen, önceki modellerin yaptığı hataları düzeltmek için sırayla eğitilir. Yüzlerce yinelemeden sonra, zayıf öğrenenler güçlü öğrenenlere dönüştürülür.

Rastgele ormanlar ve güçlendirme algoritmaları, tahmin performansını iyileştirmek için bireysel öğrenen ağaçları kullanan popüler topluluk öğrenme teknikleridir. Rastgele ormanlar, torbalama (önyükleme toplama) kavramına dayanır ve tahminlerini birleştirmek için her ağacı bağımsız olarak eğitirken, güçlendirme algoritmaları, zayıf öğrenenlerin önceki modellerin hatalarını düzeltmek için sıralı olarak eğitildiği bir katkısal yaklaşım kullanır.
## Çalışma Adımları

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
