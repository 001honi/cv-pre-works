import cv2
import numpy as np

class Dataset():
    def __init__(self, path, bgr=0):
        self.bgr    = bgr
        self.imgs   = []
        self.labels = []
        self.load_dataset(path)
        self.imgs   = np.array(self.imgs, dtype="float32")
        self.labels = np.array(self.labels)
        self.labels_one_hot = None
        self.x_train = []
        self.y_train = []
        self.x_valid = []
        self.y_valid = []

    def load_dataset(self, path):
        old_path = path # dataset/custom_dataset/
        for i in range(10):
            path = old_path + f"fig{i}/"
            self.load_figures(path, i)

    def load_figures(self, path, fig_set):
        old_path = path # dataset/custom_dataset/fig0/
        for img in range(1,201):
            path = old_path + f"img-{fig_set}-{img}.png"
            img  = cv2.imread(path, self.bgr)
            img  = cv2.resize(img, (224,224))
            self.imgs.append(img)
            self.labels.append(int(fig_set))

    def normalize(self):
        sep = 100
        size  = len(self.imgs)
        batch = int(size/sep)
        start, end = 0, batch
        for _ in range(sep):
            self.imgs[start:end] = self.imgs[start:end]/255
            start = end
            end += batch

    def shuffle(self):
        size = len(self.imgs)
        assert size == len(self.labels)
        imgs, labels = self.imgs, self.labels
        indxList = np.array(list(range(size)))
        np.random.shuffle(indxList)
        for i, indx in enumerate(indxList):
            self.imgs  [i] = imgs[indx]
            self.labels[i] = labels[indx]

    def add_channel_dim(self):
        H, W = self.imgs[0].shape
        new_shape = (len(self.imgs), H, W, 1)
        self.imgs = np.reshape(self.imgs, new_shape)
        
    def one_hot_encoding(self):
        self.labels_one_hot = []
        for label in self.labels:
            one_hot = np.zeros(10)
            one_hot[label] = 1
            self.labels_one_hot.append(one_hot)
        self.labels_one_hot = np.array(self.labels_one_hot)     

    def split_data(self, train_valid_ratio):
        size = len(self.imgs)
        valid_size = int(size//(1/train_valid_ratio))
        self.x_train = self.imgs[valid_size:]
        self.y_train = self.labels[valid_size:]
        self.x_valid = self.imgs[:valid_size]
        self.y_valid = self.labels[:valid_size]
    

 