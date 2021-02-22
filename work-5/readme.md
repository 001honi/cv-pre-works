# Ön-Çalışma-5
## Python Nesne Yönelimli Programlama | YOLO Detektörü & Tracker (Yüksek FPS)   

Bu çalışmada, COCO veri setinde eğitilmiş 80 farklı nesne tanıyan YOLOv3 modeli kullanıldı. 

Çerçeve okuma yazma için **Video** sınıfı, nesne tespiti için **YOLO** ve izleyici için **Tracker** sınıfları oluşturuldu. Bu sayede farklı bir nesne tespiti modeli uygulanmak istendiğinde yalnızca _yolo.py_ değiştirilmesi yeterli olacak.

- Aşağıdaki çıktı videosunda her 30 çerçevede bir YOLOv3 detektörü ile nesne tespiti sağlanmış ve aradaki çerçevelerde CSRT izleyicileri ile takibe devam edilmiştir.

- Bir önceki çalışmayla karşılaştırıldığında, yaklaşık 8 kat FPS artışı sağlandı.  

![YOLO_30_CSRT](videos/sample_YOLO_30_CSRT.gif) <br>

>YOLO_30_CSRT <br>
>Elapsed time: 72.333 secs <br>
>Average FPS : 10.38 <br>

- Yine bir önceki çalışmada dikkat çekilen video sonundaki kaybedilen insan, izleyiciler ile hala takip edilebildiği gözlemlendi.
