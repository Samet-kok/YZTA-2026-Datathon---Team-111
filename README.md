# YZTA 2026 Datathon - Takım 111 Çözümü

Bu depo, YZTA 2026 Datathon kapsamında Bilişsel Performans Skoru Tahmini üzerine geliştirilen en yüksek başarımlı modelleri ve bu modellerin çıktılarını içermektedir.

## Proje Hakkında
Bu çalışma, bireylerin çeşitli biyolojik ve çevresel faktörlere dayalı bilişsel performans skorlarını tahmin etmeyi amaçlayan bir regresyon problemine odaklanmaktadır. Çözüm süresince veri mühendisliği, gelişmiş topluluk öğrenme (ensemble learning) teknikleri ve optimizasyon algoritmaları kullanılarak tahmin performansı maksimize edilmiştir.

## Model Mimarileri

### 1. V21 Pipeline (Championship Pipeline)
Bu yaklaşım, model çeşitliliğini artırmak ve tahmin kararlılığını sağlamak amacıyla tasarlanmıştır.
- **Modeller:** LightGBM (L2 ve Huber Loss), XGBoost, CatBoost, HistGradientBoosting ve Derin Sinir Ağları (ANN).
- **Meta-Learner:** Tüm temel modellerin tahminleri, `BayesianRidge` meta-modeli aracılığıyla birleştirilmiştir.
- **Yöntemler:** Veri setindeki gürültüyü azaltmak amacıyla %30 ağırlıklı **Pseudo Labeling** tekniği uygulanmıştır.

### 2. V27 Pipeline (Grandmaster Pipeline)
Bu yaklaşım, CatBoost algoritmasının gücüne ve matematiksel optimizasyon tekniklerine odaklanmaktadır.
- **Modeller:** Üç farklı mutasyona (Conservative, Aggressive, Diverse) sahip CatBoost modelleri ve **RankGauss** (Quantile Transformation) uygulanmış bir Sinir Ağı.
- **Optimizasyon:** Tahminlerin ağırlıklandırılması, Out-of-Fold (OOF) skorları üzerinden **SLSQP (Sequential Least Squares Programming)** Simplex optimizasyonu ile gerçekleştirilmiştir.
- **Doğrulama:** Tahmin kararlılığını korumak adına "Target-Binned Stratified K-Fold" doğrulama stratejisi izlenmiştir.

## Kurulum ve Çalıştırma

### Gereksinimler
Gerekli kütüphaneleri yüklemek için aşağıdaki komutu kullanabilirsiniz:
```bash
pip install -r requirements.txt
```

### Kullanım
Modelleri çalıştırmak ve sonuçları (submission) üretmek için ilgili dosyayı çalıştırmanız yeterlidir:
```bash
python v21_pipeline.py
python v27_pipeline.py
```

## Dosya Yapısı
- `v21_pipeline.py`: V21 mimarisine ait eğitim ve tahmin kodu.
- `submission_v21.csv`: V21 modeli tarafından üretilen nihai tahmin dosyası.
- `v27_pipeline.py`: V27 mimarisine ait eğitim ve tahmin kodu.
- `submission_v27.csv`: V27 modeli tarafından üretilen nihai tahmin dosyası.
- `requirements.txt`: Gerekli Python kütüphaneleri.
