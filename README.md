![kredi-karti-dolandiriciligi-kobi-yasam](https://github.com/user-attachments/assets/7801fee8-dfdb-443f-bb1f-1b99524197e2)
# Fraud Detection with K-Means Clustering
Proje  dolandırıcılık tespiti ve anormallik tanımlaması için ideal olan işlemsel davranış ve finansal aktivite kalıplarına dair fraud işlemi olup olmadığı hakkında bilgi verir.README dosyasını inceleyerek  daha fazla bilgiye sahip olabilirsiniz.
# 🚀 Proje Açıklaması
Bu projede, finansal işlem verileri üzerinden şüpheli dolandırıcılık işlemlerini analiz etmek amacıyla K-Means Kümeleme (Clustering) algoritması uygulandı. Şüpheli işlemleri tespit etmek için veri kümesindeki aykırı değerler ve işlemler üzerinde detaylı analizler gerçekleştirildi. Sonuçlar, görselleştirme araçları kullanarak analiz edilerek yorumlandı ve içgörüler elde edildi.
# 🛠️ Kullanılan Teknolojiler
# Proje için aşağıdaki Python kütüphaneleri kullanıldı:
Numpy: Sayısal işlemler için kullanıldı.
Pandas: Veri işleme ve analiz işlemleri için kullanıldı.
Matplotlib ve Seaborn: Görselleştirme işlemlerinde kullanıldı.
missingno: Eksik veri analizleri için kullanıldı.
Scikit-learn: K-Means Kümeleme algoritması ve veri standardizasyonu için kullanıldı.
# 📜 İşte Kullanılan Kütüphaneler:
python
Kodu kopyala
import numpy as np               #Lineer cebir işlemleri için
import pandas as pd             # Veri işleme
import matplotlib.pyplot as sns # Görselleştirme için
import matplotlib.pyplot as plt
import seaborn as sns          # Görselleştirme için
import missingno as msno      # Eksik veri analizi için
from sklearn.cluster import KMeans  # Kümeleme algoritması
from sklearn.preprocessing import StandardScaler  # Verileri ölçeklendirme için
# 📊 Projenin Amacı
Şüpheli işlemler analiz edildi: Özellikle işlem tutarları, hesap bakiyeleri ve işlemler arasındaki korelasyon analiz edildi.
K-Means Kümeleme (Clustering) algoritması kullanılarak işlemler kümelendirildi.
İçgörüler elde edildi: Aykırı işlem desenleri ve şüpheli durumlar görselleştirilerek analiz edildi.
Görselleştirme kullanılarak analiz sonuçlarının daha sezgisel bir şekilde yorumlanması sağlandı.
# 📊 Analiz Süreci
Veri Görselleştirme: Eksik verilerin analizi ve işlem desenlerinin görselleştirilmesi yapıldı. Aykırı değerlerin etkileri grafiksel olarak incelendi.
 Keşifsel Veri Analizi(EDA)
 K-Means Kümeleme: İşlem verileri üzerinde K-Means algoritması uygulandı ve veriler kümelendirildi. Bu kümelemeler şüpheli işlem desenlerini analiz etmekte kullanıldı.
 Sonuçların Görselleştirilmesi: Seaborn ve Matplotlib kullanarak kümelerin dağılımları ve işlemler arasındaki korelasyonlar görselleştirildi.

# 📊 Çalışma Açıklaması
Adım 1: Gerekli Kütüphanelerin İçe Aktarılmasıer
Adım 2: Veri Kümesini Hazırlama
Adım 3: Görselleştirme ile Korelasyon Analizleri
Korelasyonlar ve işlem desenleri Seaborn kullanılarak grafikler aracılığıyla analiz edildi.
Adım 4: K-Means Kümeleme İşlemi
İşlem tutarları, hesap bakiyeleri gibi parametreler üzerinden K-Means Kümeleme algoritması uygulandı.

# 🎯 Elde Edilen Sonuçlar
K-Means Kümelemeleri: İşlem desenlerinin K-Means algoritması kullanarak analiz edildi ve şüpheli işlem kümeleri belirlendi.

Aykırı Değerler: Hesap bakiyeleri ve işlem tutarları üzerinden aykırı işlem örüntüleri görselleştirildi.

# Görselleştirme Sonuçları:
Sonuçlara baktığımda hesap bakiyesi düşük olan öğrencilerin işlem tutarları ve ve işlem süreleri yüksek olduğu görülmüş bunun üzlerine yaptığım derinlemesine analizler sonucu öğrenciler arası hesap bakiyesi uyumunda aykırı değerler olduğundan dolayı öğrenciler arasında  hesap bakiyesi değişkeni üzerinden  bu aykırı değerleri de kendi aralarında uyumunu analiz ettiğimde belli bir uyumda olduklarını ve kendi  aralarında herhangi bir uyumsuzluğa rastlamadığımı söyleyebilirim öğrencilerin bu durumunda  çoğunlukla yüksek işlemleri gerçekleştirirken borç yerine kredi taraflı gerçekleştirmeleri ve bu tutarların hesap bakiyesi düşük bir öğrenciye göre  çok yüksek olması şüphe uyandırıyor bunun için gerekli görselleştirmeleri  proje dosyasında yaptım.Ayrıca K-Means algoritmasını kullanarak  birbiriyle benzer işlem  tutarı ve hesap bakiyesi yönünden benzerlik gösteren 3 grup oluşturdum ve kümeleme  işlemini gerçekleştirdim.
