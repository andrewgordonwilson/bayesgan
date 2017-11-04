import os
import sys
import argparse
import json

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from bgan_util import AttributeDict
from bgan_util import print_images, MnistDataset, CelebDataset, Cifar10, SVHN, ImageNet
from bgan_models import BDCGAN

import time


def get_test_stats(session, dcgan, all_test_img_batches, all_test_lbls):

    # only need this function because bdcgan has a fixed batch size for *everything*
    # test_size is in number of batches
    all_d_logits, all_s_logits = [], []
    for test_image_batch, test_lbls in zip(all_test_img_batches, all_test_lbls):
        test_d_logits, test_s_logits = session.run([dcgan.test_D_logits, dcgan.test_S_logits],
                                                   feed_dict={dcgan.test_inputs: test_image_batch})
        all_d_logits.append(test_d_logits)
        all_s_logits.append(test_s_logits)

    test_d_logits = np.concatenate(all_d_logits)
    test_s_logits = np.concatenate(all_s_logits)
    test_lbls = np.concatenate(all_test_lbls)

    return test_d_logits, test_s_logits, test_lbls


def ml_dcgan(dataset, args):

    z_dim = args.z_dim
    x_dim = dataset.x_dim
    batch_size = args.batch_size

    print "Starting session"
    session = get_session()

    dcgan = BDCGAN(x_dim, z_dim,
                   batch_size=batch_size,
                   num_gen=1, ml=True,
                   num_classes=dataset.num_classes)

    tf.global_variables_initializer().run()

    print "Starting training loop"
        
    labeled_image_batches, label_batches = get_supervised_batches(dataset, args.N, batch_size, range(dataset.num_classes))
    test_image_batches, test_label_batches = get_test_batches(dataset, batch_size)

    for train_iter in range(args.train_iter):
        
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        image_batch, _ = dataset.next_batch(batch_size, class_id=None)
        
        rand_batch_idx = np.random.randint(len(labeled_image_batches))
        _, d_loss = session.run([dcgan.d_optim_semi, dcgan.d_loss_semi], feed_dict={dcgan.labeled_inputs: labeled_image_batches[rand_batch_idx],
                                                                                    dcgan.labels: get_gan_labels(label_batches[rand_batch_idx]),
                                                                                    dcgan.inputs: image_batch,
                                                                                    dcgan.z: batch_z})
        _, s_loss = session.run([dcgan.s_optim, dcgan.s_loss], feed_dict={dcgan.inputs: labeled_image_batches[rand_batch_idx],
                                                                          dcgan.lbls: label_batches[rand_batch_idx]})
        # compute g_sample loss
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        _, g_loss = session.run([dcgan.g_optims[0], dcgan.generation["g_losses"][0]],
                                feed_dict={dcgan.z: batch_z})

        if train_iter % args.n_save == 0:
            # get test set performance on real labels only for both GAN-based classifier and standard one
            d_logits, s_logits, lbls = get_test_stats(session, dcgan, test_image_batches, test_label_batches)
            print "saving results"
            np.savez_compressed(os.path.join(args.out_dir, 'results_%i.npz' % train_iter),
                                d_logits=d_logits, s_logits=s_logits, lbls=lbls)

            var_dict = {}
            for var in tf.trainable_variables():
                var_dict[var.name] = session.run(var.name)

            np.savez_compressed(os.path.join(args.out_dir,
                                             "weights_%i.npz" % train_iter),
                                **var_dict)
            

            print "done"

    print "closing session"
    session.close()
    tf.reset_default_graph()
