import os 
import glob
import numpy as np
import six
import cPickle
import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
import scipy.io as sio



def one_hot_encoded(class_numbers, num_classes):
    return np.eye(num_classes, dtype=float)[class_numbers]


class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))
        
        
def print_images(sampled_images, label, index, directory, save_all_samples=False):
    import matplotlib as mpl
    mpl.use('Agg') # for server side
    import matplotlib.pyplot as plt

    def unnormalize(img, cdim):
        img_out = np.zeros_like(img)
        for i in xrange(cdim):
            img_out[:, :, i] = 255.* ((img[:, :, i] + 1.) / 2.0)
        img_out = img_out.astype(np.uint8)
        return img_out
        

    if type(sampled_images) == np.ndarray:
        N, h, w, cdim = sampled_images.shape
        idxs = np.random.choice(np.arange(N), size=(5,5), replace=False)
    else:
        sampled_imgs, sampled_probs = sampled_images
        sampled_images = sampled_imgs[sampled_probs.argsort()[::-1]]
        idxs = np.arange(5*5).reshape((5,5))
        N, h, w, cdim = sampled_images.shape

        
    fig, axarr = plt.subplots(5, 5)
    for i in range(5):
        for j in range(5):
            if cdim == 1:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim)[:, :, 0], cmap="gray")
            else:
                axarr[i, j].imshow(unnormalize(sampled_images[idxs[i, j]], cdim))
            axarr[i, j].axis('off')
            axarr[i, j].set_xticklabels([])
            axarr[i, j].set_yticklabels([])
            axarr[i, j].set_aspect('equal')

    if not os.path.exists(directory):
        os.makedirs(directory)
    fig.savefig(os.path.join(directory, "%s_%i.png" % (label, index)), bbox_inches='tight')
    plt.close("all")

    if "raw" not in label.lower() and save_all_samples:
        np.savez_compressed(os.path.join(directory, "samples_%s_%i.npz" % (label, index)),
                            samples=sampled_images)
                            

            
class FigPrinter():
    
    def __init__(self, subplot_args):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig, self.ax_arr = plt.subplots(*subplot_args)
        
    def print_to_file(self, file_name, close_on_exit=True):
        import matplotlib as mpl
        mpl.use('Agg') # guarantee work on servers
        import matplotlib.pyplot as plt
        self.fig.savefig(file_name, bbox_inches='tight')
        if close_on_exit:
            plt.close("all")
        

class SynthDataset():
    
    def __init__(self, x_dim=100, num_clusters=10, seed=1234):
        
        np.random.seed(seed)
        
        self.x_dim = x_dim
        self.N = 10000
        self.true_z_dim = 2
        # generate synthetic data
        self.Xs = []
        for _ in xrange(num_clusters):
            cluster_mean = np.random.randn(self.true_z_dim) * 5 # to make them more spread
            A = np.random.randn(self.x_dim, self.true_z_dim) * 5
            X = np.dot(np.random.randn(self.N / num_clusters, self.true_z_dim) + cluster_mean,
                       A.T)
            self.Xs.append(X)
        X_raw = np.concatenate(self.Xs)
        self.X = (X_raw - X_raw.mean(0)) / (X_raw.std(0))
        print self.X.shape
        
        
    def next_batch(self, batch_size):

        rand_idx = np.random.choice(range(self.N), size=(batch_size,), replace=False)
        return self.X[rand_idx]



        
class MnistDataset():
    
    def __init__(self, data_dir):
        
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets(data_dir, one_hot=True)
        self.x_dim = [28, 28, 1]
        self.num_classes = 10
        self.dataset_size = self.mnist.train.images.shape[0]
        
    def next_batch(self, batch_size, class_id=None):
        
        if class_id is None:
            image_batch, labels = self.mnist.train.next_batch(batch_size)
            new_image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                        for n in range(image_batch.shape[0])])

            return new_image_batch, labels
        else:
            class_id_batch = np.array([])
            while class_id_batch.shape[0] < batch_size:
                image_batch, labels = self.mnist.train.next_batch(batch_size)
                image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                       for n in range(image_batch.shape[0])])
                class_id_idx = np.argmax(labels, axis=1) == class_id
                if len(class_id_idx) > 0:
                    if class_id_batch.shape[0] == 0:
                        class_id_batch = image_batch[class_id_idx]
                    else:
                        class_id_batch = np.concatenate([class_id_batch, image_batch[class_id_idx]])
            labels = np.zeros((batch_size, 10))
            labels[:, class_id] = 1.0
            return class_id_batch[:batch_size], labels

    def test_batch(self, batch_size):
        
        image_batch, labels = self.mnist.test.next_batch(batch_size)
        new_image_batch = np.array([(image_batch[n]*2. - 1.).reshape((28, 28, 1))
                                        for n in range(image_batch.shape[0])])
        return new_image_batch, labels

    def get_test_set(self):
        test_imgs = self.mnist.test.images
        test_images = np.array([(test_imgs[n]*2. - 1.).reshape((28, 28, 1))
                                for n in range(test_imgs.shape[0])])
        test_labels = self.mnist.test.labels
        return test_images, test_labels


        
        
class CelebDataset():
        
    def __init__(self, path):
        self.path = path
        self.x_dim = [32, 32, 3]

        with open(os.path.join(path, "Anno/list_attr_celeba.txt")) as af:
            lines = [line.strip() for line in af.readlines()]
    
        self.attr_dict = {}
        for bb_idx, bb_line in enumerate(lines):
            if bb_idx < 2:
                continue
            info = [token for token in bb_line.split(" ") if len(token)]
            self.attr_dict[info[0]] = [int(tk) for tk in info[1:]]

        self.salient_features = [9, 15, 20, 39] # blond, glasses, male, young
        
        self.num_classes = 2**len(self.salient_features)

        self.num_train = 75000
        self.num_test = 10000
        self.dataset_size = self.num_train


    def get_class_id(self, img_name):

        features = self.attr_dict[img_name]
        class_id = 0
        for (sfi, sf) in enumerate(self.salient_features):
            if features[sf] == 1:
                class_id += 2**sfi
        return class_id
            

    def get_batch(self, rand_idx):
        
        new_image_batch = []; new_lbl_batch = []
        for ridx in rand_idx:
            orig_name = "%06d.jpg" % (ridx + 1)
            img_name = "%06d_cropped.jpg" % (ridx + 1)
            img_path = os.path.join(self.path, "img_align_celeba/%s" % img_name)
            if not os.path.exists(img_path):
                continue
            X = imread(img_path)
            Xnorm = np.copy(X).astype(np.float64)
            Xg = np.zeros((X.shape[0], X.shape[1], 1))
            for i in xrange(3):
                Xnorm[:, :, i] /= 255.0
                Xnorm[:, :, i] = Xnorm[:, :, i] * 2. - 1.
            #Xg[:, :, 0] = 0.2126 * Xnorm[:, :, 0] + 0.7152 * Xnorm[:, :, 1] + 0.0722 * Xnorm[:, :, 2]
            new_image_batch.append(Xnorm)
            #new_image_batch.append(Xg)            

            y = self.get_class_id(orig_name)
            new_lbl_batch.append(y)

        return np.array(new_image_batch), one_hot_encoded(np.array(new_lbl_batch), self.num_classes)

    
    def next_batch(self, batch_size, class_id=None):
        got_batch = False
        while not got_batch:
            rand_idx = np.random.choice(range(self.num_train), size=(2*batch_size,), replace=False)
            X_batch, y_batch = self.get_batch(rand_idx)
            if X_batch.shape[0] >= batch_size:
                got_batch = True
                
        return X_batch[:batch_size], y_batch[:batch_size]
    

    def test_batch(self, batch_size):
        got_batch = False
        while not got_batch:
            rand_idx = np.random.choice(range(self.num_train, self.num_train + self.num_test),
                                        size=(2*batch_size,), replace=False)
            X_batch, y_batch = self.get_batch(rand_idx)
            if X_batch.shape[0] >= batch_size:
                got_batch = True
                
        return X_batch[:batch_size], y_batch[:batch_size]

    def get_test_set(self):
        return self.test_batch(1024*4)


class SVHN():

    def __init__(self, path, subsample=None):

        train_data = sio.loadmat(os.path.join(path, "train_32x32.mat"))
        test_data = sio.loadmat(os.path.join(path, "test_32x32.mat"))

        self.imgs = train_data["X"] / 255.
        self.imgs = self.imgs * 2 - 1.
        self.imgs = np.transpose(self.imgs, [3, 0, 1, 2])

        self.test_imgs = test_data["X"] / 255.
        self.test_imgs = self.test_imgs * 2 - 1.
        self.test_imgs = np.transpose(self.test_imgs, [3, 0, 1, 2])

        self.labels = np.array([yy[0]-1 for yy in train_data["y"]])
        self.labels = one_hot_encoded(self.labels, 10)

        self.test_labels = np.array([yy[0]-1 for yy in test_data["y"]])
        self.test_labels = one_hot_encoded(self.test_labels, 10)

        self.x_dim = [32, 32, 3]
        self.num_classes = 10
        self.dataset_size = self.imgs.shape[0]
        
        if subsample is not None:
            rand_idx = np.random.choice(range(self.imgs.shape[0]), 
                                        size=(int(self.imgs.shape[0]*subsample),), 
                                        replace=False)  
            self.imgs, self.labels = self.imgs[rand_idx], self.labels[rand_idx]

    def next_batch(self, batch_size, class_id=None):
        rand_idx = np.random.choice(range(self.imgs.shape[0]), size=(batch_size,), replace=False)    
        return self.imgs[rand_idx], self.labels[rand_idx]
    

    def test_batch(self, batch_size):
        rand_idx = np.random.choice(range(self.test_imgs.shape[0]),
                                    size=(batch_size,), replace=False)
        return self.test_imgs[rand_idx], self.test_labels[rand_idx]


    
def get_imagenet_val(path, x_dim, subsample=True):
    
    dirnames = [dn for dn in os.listdir(os.path.join(path, "val_256")) if dn[0] == "n"]
    assert len(dirnames), "invalid path %s given!" % (path)
    
    val_imgs = []; val_targets = []; class_dict = {}
    for dir_id, dirname in enumerate(dirnames):
        full_dirname = os.path.join(os.path.join(path, "val_256"), dirname)
        im_names = glob.glob(os.path.join(full_dirname, "*.JPEG"))
        assert len(im_names), "no images in dir %s, fix data" % full_dirname
        for im_file in im_names:
            if subsample and np.random.rand() < 0.8:
                continue
            X = imread(im_file)
            if X.shape != tuple([256, 256, 3]):
                continue
            val_imgs.append(X[None, ::4, ::4, :])
            val_targets.append(dir_id)
        class_dict[dirname] = dir_id
    
    return np.concatenate(val_imgs), np.array(val_targets), class_dict
          
    
class ImageNet():

    def __init__(self, path, num_classes, subsample=None):

        self.path = path
        self.x_dim = [64, 64, 3]
        self.num_classes = num_classes

        self.test_images, self.test_labels, self.class_dict = get_imagenet_val(self.path, self.x_dim)
        assert max(self.class_dict.values()) == self.num_classes - 1
        self.test_imgs = self.test_images / 255.
        self.test_imgs = self.test_imgs * 2 - 1.
        
        self.test_labels = one_hot_encoded(self.test_labels, self.num_classes)

        
    def supervised_batches(self, num_labeled, batch_size):

        print "generating list of supervised examples"
        dirnames = [dn for dn in os.listdir(os.path.join(self.path, "train_256")) if dn[0] == "n"]
        rand_imgs = []
        while len(rand_imgs) < num_labeled:
            rdir_name = np.random.choice(dirnames)
            rdir = os.path.join(os.path.join(self.path, "train_256"), 
                                rdir_name)
            im_names = glob.glob(os.path.join(rdir, "*.JPEG"))
            assert len(im_names), "no images in dir %s, fix data" % rdir
            rand_im_name = np.random.choice(im_names)
            if rand_im_name not in [x[1] for x in rand_imgs]:
                X = imread(rand_im_name)
                if X.shape != tuple([256, 256, 3]):
                    continue
                rand_imgs.append((rdir_name, rand_im_name))

        num_batches = num_labeled / batch_size

        while True:
            batch_id = np.random.randint(num_batches-1)
            img_batch = rand_imgs[batch_id*batch_size:(batch_id+1)*batch_size]
            batch_imgs = []; batch_lbls = []
            for rdir_name, rand_im_name in img_batch:
                batch_imgs.append(X[None, ::4, ::4, :])
                batch_lbls.append(self.class_dict[rdir_name])

            batch_images = np.concatenate(batch_imgs)
            batch_imgs = batch_images / 255.
            batch_imgs = batch_imgs * 2 - 1.

            yield (batch_imgs, one_hot_encoded(np.array(batch_lbls), self.num_classes))

            
    def next_batch(self, batch_size, class_id=None):

        dirnames = [dn for dn in os.listdir(os.path.join(self.path, "train_256")) if dn[0] == "n"]
        rdir_name = np.random.choice(dirnames)

        batch_imgs, batch_lbls, rand_imgs = [], [], []
        while len(batch_imgs) < batch_size:
            rdir = os.path.join(os.path.join(self.path, "train_256"), 
                                rdir_name)
            im_names = glob.glob(os.path.join(rdir, "*.JPEG"))
            assert len(im_names), "no images in dir %s, fix data" % rdir
            rand_im_name = np.random.choice(im_names)
            if rand_im_name not in rand_imgs:
                X = imread(rand_im_name)
                if X.shape != tuple([256, 256, 3]):
                    continue
                batch_imgs.append(X[None, ::4, ::4, :])
                batch_lbls.append(self.class_dict[rdir_name])
                rand_imgs.append(rand_im_name)
        
        self.batch_images = np.concatenate(batch_imgs)
        self.batch_imgs = self.batch_images / 255.
        self.batch_imgs = self.batch_imgs * 2 - 1.
        
        self.batch_lbls = one_hot_encoded(np.array(batch_lbls),
                                          self.num_classes)

        return self.batch_imgs, self.batch_lbls
    

    
class Cifar10():
    
    def __init__(self, path):
    

        def _convert_images(raw):
            """
            Convert images from the CIFAR-10 format and
            return a 4-dim array with shape: [image_number, height, width, channel]
            where the pixels are floats between -1.0 and 1.0.
            """
            # Convert the raw images from the data-files to floating-points.
            raw_float = (np.array(raw, dtype=float) / 255.0) * 2.0 - 1.0

            # Reshape the array to 4-dimensions.
            images = raw_float.reshape([-1, 3, 32, 32])

            # Reorder the indices of the array.
            images = images.transpose([0, 2, 3, 1])

            return images

        def process_batch(fn):
            fo = open(fn, 'rb')
            data_dict = cPickle.load(fo)
            fo.close()
            raw = data_dict["data"]
            images = _convert_images(raw)

            return images, data_dict["labels"]


        def process_meta(mfn):
            # Convert from binary strings.
            fo = open(mfn, 'rb')
            data_dict = cPickle.load(fo)
            fo.close()
            raw = data_dict["label_names"]
            names = [x.decode('utf-8') for x in raw]

            return names
                
        
        meta_name = os.path.join(path, 'batches.meta')
        self.class_names = process_meta(meta_name)
        self.num_classes = len(self.class_names)
        
        self.imgs = []
        self.labels = [] 
        for i in xrange(1, 6):
            batch_name = os.path.join(path, 'data_batch_%i' % i)
            print batch_name
            images, labels = process_batch(batch_name)
            self.imgs.append(images)
            self.labels.append(labels)
            
        self.imgs = np.concatenate(self.imgs)
        self.labels = one_hot_encoded(np.concatenate(self.labels), len(self.class_names))

        self.dataset_size = self.imgs.shape[0]
            
        test_batch_name = os.path.join(path, 'test_batch')
        print test_batch_name
        self.test_imgs, self.test_labels = process_batch(test_batch_name)
        self.test_labels = one_hot_encoded(self.test_labels, len(self.class_names))
                
        self.x_dim = [32, 32, 3]
        

    def next_batch(self, batch_size, class_id=None):
        rand_idx = np.random.choice(range(self.imgs.shape[0]), size=(batch_size,), replace=False)    
        return self.imgs[rand_idx], self.labels[rand_idx]
    

    def test_batch(self, batch_size):
        rand_idx = np.random.choice(range(self.test_imgs.shape[0]),
                                    size=(batch_size,), replace=False)
        return self.test_imgs[rand_idx], self.test_labels[rand_idx]
    
