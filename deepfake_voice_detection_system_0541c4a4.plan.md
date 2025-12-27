---
name: Deepfake Voice Detection System
overview: Kural tabanlı yöntemlerle klonlanmış sesleri tespit eden bir sistem geliştirilecek. MFCC, delta, delta-delta ve Fourier analizi kullanılarak feature extraction yapılacak ve hibrit bir skorlama sistemi ile deepfake tespiti gerçekleştirilecek.
todos:
  - id: update_sentences
    content: "record_sentences.py'yi güncelle: 20 yeni cümle ekle (Türkçe ve Arapça için)"
    status: pending
  - id: mfcc_extraction
    content: analysis.ipynb'ye MFCC feature extraction fonksiyonu ekle
    status: pending
  - id: delta_features
    content: analysis.ipynb'ye delta ve delta-delta hesaplama fonksiyonları ekle
    status: pending
    dependencies:
      - mfcc_extraction
  - id: fourier_features
    content: analysis.ipynb'ye Fourier transform tabanlı feature extraction fonksiyonları ekle (spectral centroid, rolloff, bandwidth, etc.)
    status: pending
  - id: statistical_analysis
    content: Feature'lar için istatistiksel analiz fonksiyonları ekle (mean, std, skewness, kurtosis)
    status: pending
    dependencies:
      - mfcc_extraction
      - delta_features
      - fourier_features
  - id: distance_metric
    content: Gerçek ve test sesleri arasında distance metric hesaplama fonksiyonu ekle
    status: pending
    dependencies:
      - statistical_analysis
  - id: threshold_detection
    content: Threshold tabanlı tespit fonksiyonu ekle
    status: pending
    dependencies:
      - statistical_analysis
  - id: hybrid_scoring
    content: Hibrit skorlama sistemi oluştur (tüm yöntemleri birleştirerek final skor hesapla)
    status: pending
    dependencies:
      - distance_metric
      - threshold_detection
  - id: detect_function
    content: Ana tespit fonksiyonu detect_deepfake() oluştur ve test et
    status: pending
    dependencies:
      - hybrid_scoring
  - id: visualization
    content: Feature görselleştirmeleri ekle (MFCC plots, feature distributions, skor grafikleri)
    status: pending
    dependencies:
      - detect_function
---

# Deepfa

ke Voice Detection System - Implementation Plan

## Proje Özeti

Bu proje, makine öğrenmesi kullanmadan kural tabanlı yöntemlerle klonlanmış (deepfake) sesleri tespit eden bir sistem geliştirmeyi amaçlıyor. Sistem, ses sinyallerinden MFCC, delta, delta-delta ve Fourier transform tabanlı feature'lar çıkaracak ve bunları kullanarak hibrit bir skorlama yöntemi ile deepfake tespiti yapacak.

## Mevcut Durum

- `record_sentences.py`: Şu anda 5 cümle kaydediyor (Türkçe ve Arapça)

- `analysis.ipynb`: Sadece basit waveform görselleştirme yapıyor

- `clone_real_data.py`: TTS ile ses klonlama yapıyor (XTTS v2)

- Veri yapısı: `data/real/` ve `data/cloned/` klasörleri mevcut

## Yapılacaklar

### 1. Ses Kayıt Sistemi Güncellemesi

**Dosya**: [VCR/record_sentences.py](VCR/record_sentences.py)

- Mevcut 5 cümle yerine **20 yeni cümle** eklenmeli (Türkçe ve Arapça için)

- Her kayıt 5 saniye olarak kalacak

- 2-3 kişiden kayıt alınacak (her kişi 20 kayıt)

### 2. Feature Extraction Sistemi

**Dosya**: [VCR/analysis.ipynb](VCR/analysis.ipynb)

Notebook'a şu feature extraction fonksiyonları eklenecek:

#### 2.1 MFCC Feature Extraction

- `librosa.feature.mfcc()` kullanılarak MFCC coefficients çıkarılacak

- Standart parametreler: 13 MFCC coefficient, hop_length, n_fft ayarları

#### 2.2 Delta ve Delta-Delta Hesaplama

- `librosa.feature.delta()` kullanılarak:

- Delta (1. türev): MFCC'lerin zaman içindeki değişimi

- Delta-delta (2. türev): Delta'nın değişimi

- Bu feature'lar sesin dinamik özelliklerini yakalar

#### 2.3 Fourier Transform Feature Extraction

- `numpy.fft` veya `librosa.stft()` kullanılarak:
- Spectral centroid

- Spectral rolloff

- Zero crossing rate
- Spectral bandwidth

- Mel spectrogram features

### 3. Deepfake Skor Hesaplama Sistemi (Hibrit Yöntem)

**Dosya**: [VCR/analysis.ipynb](VCR/analysis.ipynb)Hibrit skorlama sistemi şu bileşenleri içerecek:

#### 3.1 Feature Normalizasyonu

- Tüm feature'lar normalize edilecek (z-score veya min-max)

#### 3.2 İstatistiksel Analiz

- Her feature için istatistiksel özellikler (mean, std, skewness, kurtosis)

- Gerçek ve klonlanmış sesler arasındaki istatistiksel farklar

#### 3.3 Distance Metric

- Gerçek ses örnekleri ile test edilen ses arasındaki uzaklık (Euclidean, Mahalanobis, vs.)

- Eğitim setindeki gerçek seslerin feature'ları referans olarak kullanılacak

#### 3.4 Threshold Tabanlı Tespit

- Her feature için eşik değerleri belirlenecek

- Eşikleri aşan feature sayısına göre skor

#### 3.5 Hibrit Skor Hesaplama

- Tüm yöntemlerin ağırlıklı kombinasyonu

- Final skor: 0-1 arası (1 = kesinlikle sahte, 0 = kesinlikle gerçek)

### 4. Tespit Fonksiyonu

**Dosya**: [VCR/analysis.ipynb](VCR/analysis.ipynb)

```python
def detect_deepfake(audio_path, reference_real_samples=None):
    """
    Bir ses dosyasının gerçek mi sahte mi olduğunu tespit eder.
    
    Returns:
        dict: {
            'is_fake': bool,
            'score': float (0-1),
            'confidence': float,
            'feature_analysis': dict
        }
    """
```



### 5. Gerekli Kütüphaneler

**Dosya**: [VCR/requirements.txt](VCR/requirements.txt)

Mevcut kütüphaneler yeterli görünüyor, ancak eksik olabilir:

- `scikit-learn` (normalizasyon, distance metrics için)

- `scipy` (istatistiksel analiz için - zaten var)

## Veri Toplama Süreci

1. Her kişi için `record_sentences.py` çalıştırılacak

2. 20 cümle kaydedilecek (her biri 5 saniye)

3. `clone_real_data.py` ile klonlanmış versiyonlar oluşturulacak

4. Toplam: 2-3 kişi × 20 kayıt = 40-60 gerçek kayıt + 40-60 klonlanmış kayıt

## Test ve Değerlendirme

- Eğitim seti: İlk 1-2 kişinin kayıtları (feature eşiklerini belirlemek için)

- Test seti: Kalan kişinin kayıtları

- Doğruluk metrikleri: Precision, Recall, F1-score

## Notlar

- Tüm analiz `analysis.ipynb` içinde yapılacak

- Fonksiyonlar modüler olacak (tekrar kullanılabilir)