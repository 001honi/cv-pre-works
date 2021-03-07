# Ön-Çalışma-6
## Mask R-CNN ile Bölütleme (Segmentasyon)     

Bu çalışmada, COCO veri setinde eğitilmiş 80 farklı nesne tanıyan Mask R-CNN modeli kullanıldı. 

Önceki çalışmalardan farklı olarak Mask R-CNN, segmentasyon sayesinde, nesneye ait olan pikselleri de tespit etmektedir. 

- Orijinal Video Örneği

![Orijinal](videos/sample.gif) <br>

- Mask R-CNN Çıktısı

![MaskRCNN](videos/sample_MaskRCNN.gif) <br>

Colab GPU'ları ile *batch size = 1* olacak şekilde, yani her defasında GPU tek bir kare üzerinde çalışarak, çalıştırılarak elde edilen FPS sonucu:  
>MASK_RCNN <br>
>Elapsed time: 803.520 secs <br>
>Average FPS : 0.10 <br>
