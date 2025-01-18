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

### 3. Eksik Değerlerin Doldurulması

- Sayısal sütunlardaki eksik değerler medyan ile doldurulur.
- Kategorik sütunlardaki eksik değerler "Unknown" ile doldurulur:
```python
df.fillna(df.median(numeric_only=True), inplace=True)
df.fillna("Unknown", inplace=True)
```

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

### 12. Model Eğitmeyi Quality Of Live Value Üzerinden Regresyon ile Gerçekleştirme

Bu sefer target olarak Quality Of Live Catefory yerine Quality Of Live Value seçiyoruz ve devam ediyoruz.
Yine yukarıda yaptığımız işlemlerden bazılarını tekrar ediyoruz:
```python
X = df.drop(columns=["Quality of Life Value"])  # Diğer tüm özellikler
y = df["Quality of Life Value"]  # Hedef değişken
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 13. 

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
