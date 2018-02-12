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
from bgan import BDCGAN


def get_session():
    if tf.get_default_session() is None:
        print "Creating new session"
        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession()
    else:
        print "Using old session"
        _SESSION = tf.get_default_session()

    return _SESSION


def b_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size
    dataset_size = dataset.dataset_size

    session = get_session()
    tf.set_random_seed(args.random_seed)
    
    dcgan = BDCGAN(x_dim, z_dim, dataset_size, batch_size=batch_size,
                   J=args.J, J_d=args.J_d, M=args.M,
                   num_layers=args.num_layers,
                   lr=args.lr, optimizer=args.optimizer, gf_dim=args.gf_dim, 
                   df_dim=args.df_dim,
                   ml=(args.ml and args.J==1 and args.M==1 and args.J_d==1))
    
    print "Starting session"
    session.run(tf.global_variables_initializer())

    print "Starting training loop"
        
    num_train_iter = args.train_iter

    optimizer_dict = {"disc": dcgan.d_optims_adam,
                      "gen": dcgan.g_optims_adam}

    base_learning_rate = args.lr # for now we use same learning rate for Ds and Gs
    lr_decay_rate = args.lr_decay
    num_disc = args.J_d
    
    for train_iter in range(num_train_iter):

        if train_iter == 5000:
            print "Switching to user-specified optimizer"
            optimizer_dict = {"disc": dcgan.d_optims_adam,
                              "gen": dcgan.g_optims_adam}

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate *
                                                    min(1.0, (train_iter*batch_size)/float(dataset_size)))

        image_batch, _ = dataset.next_batch(batch_size, class_id=None)       

        ### compute disc losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        disc_info = session.run(optimizer_dict["disc"] + dcgan.d_losses, 
                                feed_dict={dcgan.inputs: image_batch,
                                           dcgan.z: batch_z,
                                           dcgan.d_learning_rate: learning_rate})

        d_losses = disc_info[num_disc:num_disc*2]

        ### compute generative losses
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim, dcgan.num_gen])
        gen_info = session.run(optimizer_dict["gen"] + dcgan.g_losses,
                               feed_dict={dcgan.z: batch_z,
                                          dcgan.inputs: image_batch,
                                          dcgan.g_learning_rate: learning_rate})
        g_losses = [g_ for g_ in gen_info if g_ is not None]

        if train_iter > 0 and train_iter % args.n_save == 0:

            print "Iter %i" % train_iter
            print "Disc losses = %s" % (", ".join(["%.2f" % dl for dl in d_losses]))
            print "Gen losses = %s" % (", ".join(["%.2f" % gl for gl in g_losses]))
            
            print "saving results and samples"

            results = {"disc_losses": map(float, d_losses),
                       "gen_losses": map(float, g_losses),
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
