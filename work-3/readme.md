# Ön-Çalışma-3
## CSRT & KCF ile Çoklu Nesnelerin Takibi

### v1: Seri İşleme

_tracker_v1.py_ scripti ile, seçilen takip algoritmasında (CSRT veya KCF) ve istenen _tracker_ sayısında nesne takibi yapmak mümkündür. Bu ilkel versiyonunda, _tracker_ lara ilk nesne sınırlayıcı kutuları el ile seçilerek verilmektedir. Her bir nesne için ayrı ayrı başlatılan _tracker_ lar listeye atılmakta ve okunan her frame için liste dönmektedir.   

CSRT algoritmasıyla tekil nesne takibi: (32 FPS)
 
![v1_CSRT_1](videos/race_v1_CSRT_1.gif)

CSRT algoritmasıyla çoklu nesne takibi: (8 FPS) (4 tracker için sistem performansı da 4'e bölündü)

![v1_CSRT_4](videos/race_v1_CSRT_4.gif)

KCF algoritmasıyla çoklu nesne takibinde ise saniye başına işlenen çerçeve sayısı çok daha fazla iken trackerların nesneyi kaybettiği görülüyor: (32 FPS)

![v1_KCF_4](videos/race_v1_KCF_4.gif)

### v2: Python Threading ile Paralel İşleme
