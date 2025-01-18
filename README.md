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
   - **Sınıflandırma Metriğkleri**:
     - Doğruluk (Accuracy)
     - F1 Skoru
     - ROC AUC
   - **Regresyon Metriğkleri**:
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
   - Performans metriğklerini bar grafiklerle görün.

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

Performans metriğkleri, modellerin kolay karşılaştırılması için bar grafikleri şeklinde gösterilir:

- **Sınıflandırma Metriğkleri**:
  - Doğruluk
  - F1 Skoru
  - ROC AUC

- **Regresyon Metriğkleri**:
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

Bu proje, yaşam kalitesi metriğklerini analiz etmek ve tahmin etmek amacıyla geliştirilmiştir. Geri bildirimler ve katkılar memnuniyetle karşılanır!
