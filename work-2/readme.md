# Ön-Çalışma-2
## Nesne tespiti filtreleme 

MobileNET SSD modeliyle, aşağıdaki videoda _insan_ nesneleri tespit edilmek isteniyor, yeşil sınırlayıcı kutular _insan_ nesnelerini temsil ederken istenmeyen _insan_ dışındaki tüm nesneler ise kırmızı kutular içerisinde yer alsın. _Video görüntüleri YouTube linkiyle sağlanmaktadır, **videos** klasöründe .avi formatında bulunabilir._

[![MobileNET SSD | sample-1 | out-1](videos/race_out_1.gif)](http://www.youtube.com/watch?v=pvHzxhcg104)


İstenmeyen nesneleri etiketleri üzerinden eleyebileceğimiz gibi tespit _confidence_ değerini yükselterek de modelin emin olamadığı tespitleri eleyebiliriz. Ancak burada bir trade-off olup bu değer yüksek tutulduğunda ise _insan_ nesnelerini kaçırabilmemiz söz konusudur. Bu noktada, modeli _tracking_ algoritmaları ile desteklemek en iyi çözüme ulaşmamızı sağlayabilir.

[![MobileNET SSD | sample-1 | out-2](videos/race_out_2.gif)](http://www.youtube.com/watch?v=Rt-f-1R0pYY)

## Detektörü Google Colab üzerinden Tesla T4 GPU ile ivmelendirme

Önceki çalışmada tek bir çerçevede nesne tespiti yapılmıştı; video üzerinde çalışıldığında ise CPU donanımları yetersiz kalabilmektedir. 

Bu ön çalışmada kullanılan video örneğinin, benim bilgisayarımda Intel Core i7-7500U CPU'su ile işlenmesi yaklaşık olarak saniyede 26 çerçeve (26 FPS) hız ile gerçekleşmektedir. İşlenen görüntünün anında ekranda gösterilmesi istendiğinde ise işlem hızı 21 FPS'ye düşmektedir. 
>[INFO] accessing video stream...<br>
[INFO] elapsed time: 13.11<br>
[INFO] approx. FPS: 25.93<br>

Aynı işlemi Google Colab CPU'su ile çalıştırdığımda 17 FPS elde ettim.
>!python ssd_detector_with_gpu_acc.py --use-gpu False

>[INFO] accessing video stream... <br>
[INFO] elapsed time: 19.56 <br>
[INFO] approx. FPS: 17.38 <br>

Google Colab sunucularındaki standart OpenCV kütüphanesi versiyonu, (v4.4), Tesla T4 GPU'ları ile çalışmak istendiğinde hata vermektedir.
>[INFO] setting preferable backend and target to CUDA...<br>
Traceback (most recent call last):<br>
  File "ssd_detector_gpu.py", line 44, in <module>  net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)<br>
AttributeError: module 'cv2.dnn' has no attribute 'DNN_BACKEND_CUDA'
  
Uyumsuzluk problemi, OpenCV'nin resmi olmayan bir versiyonuna güncellenerek giderilebiliyor; ancak, bu güncelleme için OpenCV'nin yeniden derlenmesi gerekmekte ve bu işlem oldukça zaman almakta. İlerleyen süreçte bir düzenleme olacağına inanıyorum.
>%cd /content <br>
!git clone https://github.com/opencv/opencv <br>
!git clone https://github.com/opencv/opencv_contrib <br>
!mkdir /content/build <br>
%cd /content/build <br>
!cmake -DOPENCV_EXTRA_MODULES_PATH=/content/opencv_contrib/modules  -DBUILD_SHARED_LIBS=OFF  -DBUILD_TESTS=OFF  -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_OPENEXR=OFF -DWITH_CUDA=ON -DWITH_CUBLAS=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON /content/opencv <br>
!make -j8 install

Uzun bir derleme süresinden sonra programı GPU ile çalıştırdığımda beklenmedik bir sonuçla karşılaştım.
>!python ssd_detector_with_gpu_acc.py --use-gpu True

>[INFO] setting preferable backend and target to CUDA... <br>
[INFO] accessing video stream... <br>
[INFO] elapsed time: 26.47 <br>
[INFO] approx. FPS: 12.85 <br>

CUDA çekirdekleriyle ivmelendirilmiş süreç, işlemci performansından daha kötü bir sonuç verdi. Colab notebook ile drive bağlantısında sorun yaşanmış olabilir veya uyguladığım derleme eskimiş olabilirdi. (Ekim, 2020) Küçük bir araştırma yaptığımda bu derleme üzerine yeni resmi güncellemeler geldiğini öğrendim.

Bu konu üzerine yazılmış daha kapsamlı bloglar buldum; yapılacaklar listesine eklendi.

