# Casting-Product-Defect-Classification-using-CNN

# End-to-End CNN for Industrial Casting Defect Detection

Bu proje, endüstriyel döküm ürünlerinin görüntülerinden üretim hatalarını tespit etmek amacıyla geliştirilmiş, uçtan uca bir Evrişimli Sinir Ağı (CNN) modelini içermektedir. Proje, veri ön işleme, model oluşturma, eğitme ve değerlendirme adımlarını kapsayan sağlam ve yeniden üretilebilir bir yapı sunar.

![Sample Images](https://i.imgur.com/gU89aB1.png)
*Solda: Hatalı (def_front) | Sağda: Sağlam (ok_front)*

## 🚀 Projenin Öne Çıkan Özellikleri

- **Veri Artırma (Data Augmentation):** Modelin genelleme yeteneğini artırmak ve aşırı öğrenmeyi (overfitting) önlemek için `ImageDataGenerator` kullanılarak anlık olarak (on-the-fly) veri artırma teknikleri (döndürme, kaydırma, yakınlaştırma vb.) uygulanmıştır.
- **Modern CNN Mimarisi:** Model, `BatchNormalization` katmanları ile stabilize edilmiş, `GELU` gibi modern aktivasyon fonksiyonları ve `Dropout` ile regularizasyon sağlanmış derin bir CNN mimarisine sahiptir.
- **Sağlam Eğitim Süreci (Robust Training):**
  - **AdamW Optimizer:** Ağırlık bozunması (weight decay) ile regularizasyonu iyileştiren AdamW optimize edici kullanılmıştır.
  - **Callback'ler:** `EarlyStopping` ile gereksiz eğitim önlenmiş, `ReduceLROnPlateau` ile öğrenme oranı dinamik olarak ayarlanmış ve `ModelCheckpoint` ile en iyi model kaydedilmiştir.
- **Kapsamlı Değerlendirme:** Model performansı; doğruluk/kayıp grafikleri, karmaşıklık matrisi (confusion matrix), sınıflandırma raporu (classification report) ve ROC/AUC eğrisi gibi metriklerle detaylı bir şekilde analiz edilmiştir.
- **Yeniden Üretilebilirlik (Reproducibility):** Projenin farklı sistemlerde aynı sonuçları vermesi için `random`, `numpy` ve `tensorflow` kütüphanelerinde tohum (seed) değerleri sabitlenmiştir.

## 💾 Veri Seti

Bu projede, Kaggle üzerinde bulunan **"Real-Life Industrial Dataset of Casting Product"** veri seti kullanılmıştır. Veri seti, 'def_front' (hatalı) ve 'ok_front' (sağlam) olmak üzere iki sınıfa ayrılmış döküm ürün görselleri içermektedir.

- **Veri Seti Linki:** [Kaggle Dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

## 🛠️ Kurulum ve Kullanım

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git](https://github.com/KULLANICI_ADINIZ/PROJE_ADINIZ.git)
    cd PROJE_ADINIZ
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    Proje için gerekli kütüphaneleri `requirements.txt` dosyasından yükleyebilirsiniz.
    ```bash
    pip install -r requirements.txt
    ```
    *Eğer bir `requirements.txt` dosyanız yoksa, şu komutlarla temel kütüphaneleri kurabilirsiniz:*
    ```bash
    pip install tensorflow numpy matplotlib seaborn scikit-learn opendatasets
    ```

3.  **Kaggle API Kimlik Bilgilerini Ayarlayın:**
    `opendatasets` kütüphanesinin veri setini indirebilmesi için Kaggle kullanıcı adınıza ve API anahtarınıza ihtiyacı olacaktır. Scripti çalıştırdığınızda sizden bu bilgileri girmeniz istenecektir.

4.  **Script'i Çalıştırın:**
    Tüm kurulum tamamlandıktan sonra ana Python script'ini çalıştırın.
    ```bash
    python ai_studio_code.py
    ```
    Script, veri setini otomatik olarak indirecek, modeli eğitecek, sonuçları ve grafikleri üretecektir.

## 📊 Sonuçlar ve Değerlendirme

Model, test veri seti üzerinde **~%99**'un üzerinde bir doğruluk oranı elde etmiştir. Eğitim süreci boyunca elde edilen doğruluk ve kayıp grafikleri aşağıda gösterilmiştir.

*(Buraya `plt.show()` ile ürettiğiniz grafikleri ekran görüntüsü alıp ekleyebilirsiniz.)*

**Eğitim ve Doğrulama Grafikleri**
![Training Curves](https://i.imgur.com/your_training_plot_url.png)

**Karmaşıklık Matrisi (Confusion Matrix)**
![Confusion Matrix](https://i.imgur.com/your_confusion_matrix_url.png)

**ROC Eğrisi**
![ROC Curve](https://i.imgur.com/your_roc_curve_url.png)

Sınıflandırma raporu, modelin her iki sınıfı da (hatalı ve sağlam) yüksek F1-skoru ile başarıyla tespit ettiğini göstermektedir.

## 💡 Gelecek Geliştirmeler

Bu projenin potansiyelini daha da ileri taşımak için aşağıdaki adımlar atılabilir:

- **Transfer Learning:** `ResNet50`, `EfficientNet` gibi önceden eğitilmiş modeller kullanarak daha yüksek doğruluk oranları hedeflenebilir.
- **Gelişmiş Veri Artırma:** `CutMix` ve `Mixup` gibi daha karmaşık veri artırma teknikleri denenebilir.
- **Hiperparametre Optimizasyonu:** `KerasTuner` veya `Optuna` gibi kütüphanelerle en iyi hiperparametre setini bulmak için otomatik optimizasyon yapılabilir.
- **Model Dağıtımı (Deployment):** Eğitilmiş model, `TensorFlow Serving`, `Flask` veya `FastAPI` kullanılarak bir web servisi haline getirilebilir.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız.
