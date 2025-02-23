# Oral Kanser Sınıflandırma Modeli

Bu proje, **derin öğrenme kullanarak oral kanser tespiti** yapmayı amaçlayan bir sınıflandırma modelidir. TensorFlow ve Keras kullanılarak geliştirilmiş olup, görüntü işleme teknikleri ile desteklenmiştir.

## Kullanılan Teknolojiler
- TensorFlow & Keras
- Matplotlib
- ImageDataGenerator (Veri artırma)
- Convolutional Neural Networks (CNN)

## Veri Kümesi
Bu model, **Oral Kanser** görüntüleri içeren bir veri kümesi üzerinde eğitilmiştir. Veriler eğitim ve doğrulama olarak ikiye ayrılmıştır:
- **%90 Eğitim**
- **%10 Doğrulama**

## Model Mimarisi
- 2 adet **Conv2D** katmanı
- **MaxPooling2D** ile boyut küçültme
- **Flatten** katmanı
- **128 nöronlu Dense katmanı**
- **Sigmoid aktivasyonlu çıkış katmanı** (İkili sınıflandırma için)

## Eğitim
Model **Adam optimizasyon algoritması** kullanılarak derlenmiş ve `binary_crossentropy` kayıp fonksiyonu ile eğitilmiştir. **10 epoch boyunca** eğitim gerçekleştirilmiştir.

## Sonuç Görselleştirme
Eğitim sürecinin analizini yapmak için doğruluk ve kayıp grafikleri çizdirilmektedir.

## Örnek Çıktılar
Modelin tahmin gücünü görmek için doğrulama setinden bazı örnekler görselleştirilmiştir. Her resim için **tahmin edilen ve gerçek sınıf** gösterilmektedir.

---

# Oral Cancer Classification Model

This project aims to detect **oral cancer using deep learning**. It is developed using TensorFlow and Keras and enhanced with image processing techniques.

## Technologies Used
- TensorFlow & Keras
- Matplotlib
- ImageDataGenerator (Data Augmentation)
- Convolutional Neural Networks (CNN)

## Dataset
This model is trained on a dataset containing **Oral Cancer** images. The data is split into training and validation sets:
- **90% Training**
- **10% Validation**

## Model Architecture
- 2 **Conv2D** layers
- **MaxPooling2D** for dimensionality reduction
- **Flatten** layer
- **128-neuron Dense layer**
- **Sigmoid activation output layer** (for binary classification)

## Training
The model is compiled using the **Adam optimization algorithm** and trained with the `binary_crossentropy` loss function. The training is conducted for **10 epochs**.

## Visualization of Results
Accuracy and loss graphs are plotted to analyze the training process.

## Sample Outputs
To evaluate the model's prediction power, some examples from the validation set are visualized. Each image displays both the **predicted and actual class**.

