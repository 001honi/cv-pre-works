# Ön-Çalışma-4
## YOLO Detektörü (Düşük FPS)

Bu çalışmada, COCO veri setinde eğitilmiş 80 farklı nesne tanıyan YOLOv3 modeli kullanıldı. 

_yolo_detector.py_ scripti, verilen video içerisindeki her frame ile YOLO modelini beslemekte çıkışlarını işaretlemektedir. İşlemci üzerinde çalıştığından oldukça düşük bir FPS değeri elde edilmektedir. 

- Toplamda 751 çerçeve içeren 25 saniyelik videonun işlenmesi 572 saniye sürmekte ve bu süre bize 1.31 FPS değerini vermektedir. 

[![](http://img.youtube.com/vi/THE7GwAWriU/0.jpg)](http://www.youtube.com/watch?v=THE7GwAWriU "YOLO")

>YOLO <br>
>Elapsed time: 572.554 secs <br>
>Average FPS : 1.31 <br>

- Video sonuna doğru detektör insan nesnesini algılayamıyor; burada bir _tracking_ gerçekleşirse sınırlayıcı kutu bisiklet üzerindeki insanı izlemeye devam edecektir.


<hr>

Her çerçeve için YOLO detektörü çalıştırmak yerine, N çerçevede bir detektör çalıştırılıp üretilen sınırlayıcı kutular ile _tracker_ lar başlatılırsa işlem maliyetini kısabiliriz. Ancak bu deneme için oluşturduğum _yolo_kcf_v0.1.py_ scriptinde _tracker_ ları multiprocessing ile kontrol etmek istediğimde _memory allocation_ problemiyle karşılaştım. 

Bir sonraki ön çalışmada, daha stabil ve daha genellenebilir çalışacak özgün projemi yazacağım.
