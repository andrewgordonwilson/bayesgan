#!/usr/bin/env python

import os
import sys
import argparse
import json
import time

import numpy as np
from math import ceil

from PIL import Image

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, SVHN, ImageNet
from bgan_semi import BDCGAN_Semi

def get_session():
    if tf.get_default_session() is None:
        print "Creating new session"
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print "Using old session"
        _SESSION = tf.get_default_session()

    return _SESSION


def get_gan_labels(lbls):
    # add class 0 which is the "fake" class
    if lbls is not None:
        labels = np.zeros((lbls.shape[0], lbls.shape[1] + 1))
        labels[:, 1:] = lbls
    else:
        labels = None
    return labels


def get_supervised_batches(dataset, size, batch_size, class_ids):

    def batchify_with_size(sampled_imgs, sampled_labels, size):
        rand_idx = np.random.choice(range(sampled_imgs.shape[0]), size, replace=False)
        imgs_ = sampled_imgs[rand_idx]
        lbls_ = sampled_labels[rand_idx]
        rand_idx = np.random.choice(range(imgs_.shape[0]), batch_size, replace=True)
        imgs_ = imgs_[rand_idx]
        lbls_ = lbls_[rand_idx] 
        return imgs_, lbls_

    labeled_image_batches, lblss = [], []
    num_passes = int(ceil(float(size) / batch_size))
    for _ in xrange(num_passes):
        for class_id in class_ids:
            labeled_image_batch, lbls = dataset.next_batch(int(ceil(float(batch_size)/len(class_ids))),
                                                           class_id=class_id)
            labeled_image_batches.append(labeled_image_batch)
            lblss.append(lbls)

    labeled_image_batches = np.concatenate(labeled_image_batches)
    lblss = np.concatenate(lblss)

    if size < batch_size:
        labeled_image_batches, lblss = batchify_with_size(labeled_image_batches, lblss, size)

    shuffle_idx = np.arange(lblss.shape[0]); np.random.shuffle(shuffle_idx)
    labeled_image_batches = labeled_image_batches[shuffle_idx]
    lblss = lblss[shuffle_idx]

    while True:
        i = np.random.randint(max(1, size/batch_size))
        yield (labeled_image_batches[i*batch_size:(i+1)*batch_size],
               lblss[i*batch_size:(i+1)*batch_size])


def get_test_batches(dataset, batch_size):

    try:
        test_imgs, test_lbls = dataset.test_imgs, dataset.test_labels
    except:
        test_imgs, test_lbls = dataset.get_test_set()

    all_test_img_batches, all_test_lbls = [], []
    test_size = test_imgs.shape[0]
    i = 0
    while (i+1)*batch_size <= test_size:
        all_test_img_batches.append(test_imgs[i*batch_size:(i+1)*batch_size])
        all_test_lbls.append(test_lbls[i*batch_size:(i+1)*batch_size])
        i += 1

    return all_test_img_batches, all_test_lbls



def get_test_accuracy(session, dcgan, all_test_img_batches, all_test_lbls):

    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_probs, all_s_probs = [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_probs, test_s_probs = session.run([dcgan.test_d_probs, dcgan.test_s_probs],
                                                   feed_dict={dcgan.test_inputs: test_image_batch})
        ensemble_d_probs = np.concatenate([d_probs_[:, :, None] for d_probs_ in test_d_probs], axis=-1).sum(-1)
        all_d_probs.append(ensemble_d_probs)
        all_s_probs.append(test_s_probs)

    test_d_probs = np.concatenate(all_d_probs)
    test_s_probs = np.concatenate(all_s_probs)
    test_lbls = np.concatenate(all_test_lbls)

    sup_acc = (100. * np.sum(np.argmax(test_s_probs, 1) == np.argmax(test_lbls, 1)))\
              / test_lbls.shape[0]
        
    semi_sup_acc = (100. * np.sum(np.argmax(test_d_probs, 1) == np.argmax(test_lbls, 1)))\
              / test_lbls.shape[0]
        
    print "Sup acc: %.5f" % (sup_acc)
    print "Semi-sup acc: %.5f" % (semi_sup_acc)

    return sup_acc, semi_sup_acc
    


def b_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size
    dataset_size = dataset.dataset_size

    session = get_session()
    tf.set_random_seed(args.random_seed)

    dcgan = BDCGAN_Semi(x_dim, z_dim, dataset_size, batch_size=batch_size, J=args.J, J_d=args.J_d, M=args.M,
                        num_layers=args.num_layers,
                        lr=args.lr, optimizer=args.optimizer, gf_dim=args.gf_dim, 
                        df_dim=args.df_dim, ml=(args.ml and args.J==1 and args.M==1 and args.J_d==1),
                        num_classes=dataset.num_classes)
    
    print "Starting session"
    session.run(tf.global_variables_initializer())

    print "Starting training loop"
        
    num_train_iter = args.train_iter

    if hasattr(dataset, "supervised_batches"):
        # implement own data feeder if data doesnt fit in memory
        supervised_batches = dataset.supervised_batches(args.N, batch_size)
    else:
        supervised_batches = get_supervised_batches(dataset, args.N, batch_size, range(dataset.num_classes))

    test_image_batches, test_label_batches = get_test_batches(dataset, batch_size)

    optimizer_dict = {"disc_semi": dcgan.d_optims_semi_adam,
                      "sup_d": dcgan.s_optim_adam,
                      "gen": dcgan.g_optims_semi_adam}

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay
    num_disc = args.J_d
    
    for train_iter in range(num_train_iter):

        if train_iter == 5000:
            print "Switching to user-specified optimizer"
            optimizer_dict = {"disc_semi": dcgan.d_optims_semi,
                              "sup_d": dcgan.s_optim,
                              "gen": dcgan.g_optims_semi}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))

        image_batch, _ = dataset.next_batch(batch_size, class_id=None)       
        labeled_image_batch, labels = supervised_batches.next()

        ### compute disc losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        disc_info = session.run(optimizer_dict["disc_semi"] + dcgan.d_losses,
                                feed_dict={dcgan.labeled_inputs: labeled_image_batch,
                                           dcgan.labels: labels,
                                           dcgan.inputs: image_batch,
                                           dcgan.z: batch_z,
                                           dcgan.d_semi_learning_rate: learning_rate})
        d_losses = [d_ for d_ in disc_info if d_ is not None]

        ### compute generative losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        gen_info = session.run(optimizer_dict["gen"] + dcgan.g_losses,
                               feed_dict={dcgan.z: batch_z,
                                          dcgan.inputs: image_batch,
                                          dcgan.g_learning_rate: learning_rate})
        g_losses = [g_ for g_ in gen_info if g_ is not None]

        ### vanilla supervised loss
        _, s_loss = session.run([optimizer_dict["sup_d"], dcgan.s_loss], feed_dict={dcgan.inputs: labeled_image_batch,
                                                                                    dcgan.lbls: labels})

        if train_iter > 0 and train_iter % args.n_save == 0:

            print "Iter %i" % train_iter
            print "Disc losses = %s" % (", ".join(["%.2f" % dl for dl in d_losses]))
            print "Gen losses = %s" % (", ".join(["%.2f" % gl for gl in g_losses]))
            
            # get test set performance on real labels only for both GAN-based classifier and standard one
            s_acc, ss_acc = get_test_accuracy(session, dcgan, test_image_batches, test_label_batches)
            print "Sup classification acc: %.2f" % (s_acc)
            print "Semi-sup classification acc: %.2f" % (ss_acc)

            print "saving results and samples"

            results = {"disc_losses": map(float, d_losses),
                       "gen_losses": map(float, g_losses),
                       "supervised_acc": float(s_acc),
                       "semi_supervised_acc": float(ss_acc),
                       "timestamp": time.time()}

            with open(os.path.join(args.out_dir, 'results_%i.json' % train_iter), 'w') as fp:
                json.dump(results, fp)
            
            if args.save_samples:
                for zi in xrange(dcgan.num_gen):
                    _imgs, _ps = [], []
                    for _ in range(10):
                        z_sampler = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                        sampled_imgs = session.run(dcgan.gen_samplers[zi*dcgan.num_mcmc],
                                                   feed_dict={dcgan.z_sampler: z_sampler})
                        _imgs.append(sampled_imgs)
                    sampled_imgs = np.concatenate(_imgs)
                    print_images(sampled_imgs, "B_DCGAN_%i_%.2f" % (zi, g_losses[zi*dcgan.num_mcmc]),
                                 train_iter, directory=args.out_dir)

                print_images(image_batch, "RAW", train_iter, directory=args.out_dir)

            if args.save_weights:
                var_dict = {}
                for var in tf.trainable_variables():
                    var_dict[var.name] = session.run(var.name)

                np.savez_compressed(os.path.join(args.out_dir,
                                                 "weights_%i.npz" % train_iter),
                                    **var_dict)
            

            print "done"
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN experiments')

    parser.add_argument('--out_dir',
                        type=str,
                        required=True,
                        help="location of outputs (root location, which exists)")

    parser.add_argument('--n_save',
                        type=int,
                        default=100,
                        help="every n_save iteration save samples and weights")
    
    parser.add_argument('--z_dim',
                        type=int,
                        default=100,
                        help='dim of z for generator')
    
    parser.add_argument('--gf_dim',
                        type=int,
                        default=64,
                        help='num of gen features')
    
    parser.add_argument('--df_dim',
                        type=int,
                        default=96,
                        help='num of disc features')
    
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='path to where the datasets live')

    parser.add_argument('--dataset',
                        type=str,
                        default="mnist",
                        help='datasate name mnist etc.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="minibatch size")

    parser.add_argument('--prior_std',
                        type=float,
                        default=1.0,
                        help="NN weight prior std.")

    parser.add_argument('--num_layers',
                        type=int,
                        default=4,
                        help="number of layers for G and D nets")

    parser.add_argument('--num_gen',
                        type=int,
                        dest="J",
                        default=1,
                        help="number of samples of z/generators")

    parser.add_argument('--num_disc',
                        type=int,
                        dest="J_d",
                        default=1,
                        help="number of discrimitor weight samples")

    parser.add_argument('--num_mcmc',
                        type=int,
                        dest="M",
                        default=1,
                        help="number of MCMC NN weight samples per z")

    parser.add_argument('--N',
                        type=int,
                        default=128,
                        help="number of supervised data samples")

    parser.add_argument('--train_iter',
                        type=int,
                        default=50000,
                        help="number of training iterations")

    parser.add_argument('--wasserstein',
                        action="store_true",
                        help="wasserstein GAN")

    parser.add_argument('--ml',
                        action="store_true",
                        help="if specified, disable bayesian things")

    parser.add_argument('--save_samples',
                        action="store_true",
                        help="wether to save generated samples")
    
    parser.add_argument('--save_weights',
                        action="store_true",
                        help="wether to save weights")

    parser.add_argument('--random_seed',
                        type=int,
                        default=2222,
                        help="random seed")
    
    parser.add_argument('--lr',
                        type=float,
                        default=0.005,
                        help="learning rate")

    parser.add_argument('--lr_decay',
                        type=float,
                        default=3.0,
                        help="learning rate")

    parser.add_argument('--optimizer',
                        type=str,
                        default="sgd",
                        help="optimizer --- 'adam' or 'sgd'")

    
    args = parser.parse_args()

    # set seeds
    np.random.seed(args.random_seed)
    tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print "Creating %s" % args.out_dir
        os.makedirs(args.out_dir)
    args.out_dir = os.path.join(args.out_dir, "bgan_%s_%i" % (args.dataset, int(time.time())))
    os.makedirs(args.out_dir)

    import pprint
    with open(os.path.join(args.out_dir, "hypers.txt"), "w") as hf:
        hf.write("Hyper settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
        
    celeb_path = os.path.join(args.data_path, "celebA")
    cifar_path = os.path.join(args.data_path, "cifar-10-batches-py")
    svhn_path = os.path.join(args.data_path, "svhn")
    mnist_path = os.path.join(args.data_path, "mnist") # can leave empty, data will self-populate
    imagenet_path = os.path.join(args.data_path, args.dataset)
    #imagenet_path = os.path.join(args.data_path, "imagenet")

    if args.dataset == "mnist":
        dataset = MnistDataset(mnist_path)
    elif args.dataset == "celeb":
        dataset = CelebDataset(celeb_path)
    elif args.dataset == "cifar":
        dataset = Cifar10(cifar_path)
    elif args.dataset == "svhn":
        dataset = SVHN(svhn_path)
    elif "imagenet" in args.dataset:
        num_classes = int(args.dataset.split("_")[-1])
        dataset = ImageNet(imagenet_path, num_classes)
    else:
        raise RuntimeError("invalid dataset %s" % args.dataset)

    ### main call
    b_dcgan(dataset, args)
