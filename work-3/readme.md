# Ön-Çalışma-3
## CSRT & KCF ile Çoklu Nesnelerin Takibi

### v1: Seri İşleme

_tracker_v1.py_ scripti ile, seçilen takip algoritmasında (CSRT veya KCF) ve istenen _tracker_ sayısında nesne takibi yapmak mümkündür. Bu ilkel versiyonunda, _tracker_ lara ilk nesne sınırlayıcı kutuları el ile seçilerek verilmektedir. Her bir nesne için ayrı ayrı başlatılan _tracker_ lar listeye atılmakta ve okunan her frame için liste dönmektedir.   

> CSRT algoritmasıyla tekil nesne takibi: (**32 FPS**)
 
![v1_CSRT_1](videos/race_v1_CSRT_1.gif)

> CSRT algoritmasıyla çoklu nesne takibi: (**8 FPS**) (4 tracker için sistem performansı da 4'e bölündü)

![v1_CSRT_4](videos/race_v1_CSRT_4.gif)

> KCF algoritmasıyla çoklu nesne takibinde ise saniye başına işlenen çerçeve sayısı çok daha fazla iken trackerların nesneyi kaybettiği görülüyor: (**32 FPS**)

![v1_KCF_4](videos/race_v1_KCF_4.gif)

<hr>

### v2: Python 'Multiprocessing' Kütüphanesiyle Paralel İşleme

_tracker_v2.py_ scripti ile çoklu nesne takipçisine paralel işlem yapma yeteneği kazandırıldı. 

> CSRT algoritmasıyla işlemcinin tek çekirdeğiyle 4 nesne takibi yapılan önceki seri işleme durumunda 8 FPS hız elde edilirken, süreç 4 çekirdeğe dağıtıldığında kayda değer bir hızlanma görülmektedir. (**14 FPS**)

![v2_CSRT_4](videos/race_v2_CSRT_4.gif)
