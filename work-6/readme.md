# Ön-Çalışma-6
## Mask R-CNN ile Bölütleme (Segmentasyon)     

Bu çalışmada, COCO veri setinde eğitilmiş 80 farklı nesne tanıyan Mask R-CNN modeli kullanıldı. 

Önceki çalışmalardan farklı olarak Mask R-CNN, segmentasyon sayesinde, nesneye ait olan pikselleri de tespit etmektedir. 

- Orijinal Video Örneği

![Orijinal](videos/sample.gif) <br>

- Mask R-CNN Çıktısı

![MaskRCNN](videos/sample_MaskRCNN.gif) <br>

- Segmentasyon daha yüksek işlem gücü gerektirdiğinden, saniye başına işlenen çerçeve sayısında düşüş gözlemlenmekte.  
>MASK_RCNN <br>
>Elapsed time: 932.510 secs <br>
>Average FPS : 0.09 <br>
