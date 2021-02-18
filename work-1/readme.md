# Ön-Çalışma-1
## SSD Detektörü ve IoU performans ölçümü

MobileNET SSD, mobil cihazlarda çalışabilecek seviyede küçük boyutlu ve hızlı bir mimari olup COCO veri seti üzerinde eğitilmiş [Caffe modeli](https://github.com/chuanqi305/MobileNet-SSD), OpenCV'nin _dnn_ modülüyle Python ortamına alındı. 


Bu model, 'background' dışında 20 farklı nesneyi tanıyabiliyor: <br>
["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable",	"dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]

Beyoğlu'nda çekilmiş bir görseli SSD detektörümüze giriş olarak verdiğimizde, aşağıdaki çıktıyı vermektedir. Tren için otobüs tahmininde bulunması hayal kırıklığı yaratsa da bu çalışmanın asıl amacı yüksek başarım elde etmek değil; modelin performansını 'Intersection over Union' metriği ile hesaplamaktır.
<p align="center">
  <img src="https://github.com/001honi/cv-pre-works/blob/main/work-1/images/beyoglu_out.jpg" />
</p>  

_DarkLabel_ programı üzerinden el ile _ground truth_ etiketleme gerçekleştirildi ve text formatında düzenlendi. (gt_beyoglu.txt) <br>
Benzer şekilde _ssd_detector.py_ scriptinde elde edilen tahmin sonuçları da aynı formatta yazdırıldı. (pr_beyoglu.txt) <br>

IoU hesaplanırken, etiketlenen ve tahmin ile üretilen sınırlayıcı kutu koordinatları aynı sırada bulunmayabilir; bunun için x1 koordinatları üzerinden iki liste de yeniden sıralandı. Ancak, detektörün farkına varmadığı nesneler tüm hesapların kaymasına neden olabilir. Bu sorunun üstesinden gelmek için daha karmaşık bir algoritma gerekmektedir, daha sonraki çalışmalarımda buna çözüm arayacağım.

0.5 puan üzeri başarılı olarak nitelendiriliyor; aşağıda konsola yazdırılan IoU puanları bulunmaktadır. <br>
> Label: person     IoU Score: 0.564<br>
Label: person     IoU Score: 0.687<br>
Label: person     IoU Score: 0.044<br>
Label: person     IoU Score: 0.490<br>
Label: bus        IoU Score: 0.857<br>
Label: person     IoU Score: 0.384<br>
Label: person     IoU Score: 0.634<br>

Anlaşılabilirliği artırmak için _ground truth_ bölgesi alpha=0.5 olacak şekilde transparan boyandı.
<p align="center">
  <img src="https://github.com/001honi/cv-pre-works/blob/main/work-1/images/beyoglu_iou.jpg" />
</p> 
