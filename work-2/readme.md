# Ön-Çalışma-2
## Nesne tespiti filtreleme & Detektörü Google Colab üzerinden Tesla T4 GPU ile ivmelendirme

Önceki çalışmada tek bir çerçevede nesne tespiti yapılmıştı; video üzerinde çalışıldığında ise CPU donanımları yetersiz kalabilmektedir. 

Bu ön çalışmada kullanılan video örneğinin, benim bilgisayarımda Intel Core i7-7500U CPU'su ile işlenmesi yaklaşık olarak saniyede 26 çerçeve (26 FPS) hız ile gerçekleşmektedir. İşlenen görüntünün anında ekranda gösterilmesi istendiğinde ise işlem hızı 21 FPS'ye düşmektedir. 

Aşağıdaki videoda _insan_ nesneleri tespit edilmek isteniyor, yeşil sınırlayıcı kutular _insan_ nesnelerini temsil ederken istenmeyen _insan_ dışındaki tüm nesneler ise kırmızı kutular içerisinde yer alsın.


<!-- blank line -->
<figure class="video_container">
  <iframe src="https://www.youtube.com/embed/enMumwvLAug" frameborder="0" allowfullscreen="true"> </iframe>
</figure>
<!-- blank line -->

İstenmeyen nesneleri etiketleri üzerinden eleyebileceğimiz gibi tespit _confidence_ değerini yükselterek de modelin emin olamadığı tespitleri eleyebiliriz. Ancak burada bir trade-off olup bu değer yüksek tutulduğunda ise _insan_ nesnelerini kaçırabilmemiz söz konusudur. Burada, modeli _tracking_ algoritmaları ile desteklemek en iyi çözüme ulaşmamızı sağlayabilir.

