#!/usr/bin/env python

import os
import sys

import tensorflow as tf
import numpy as np

from bgan_models import BGAN
from bgan_util import SynthDataset, FigPrinter

from sklearn import mixture


def get_session():
    global _SESSION
    if tf.get_default_session() is None:
        _SESSION = tf.InteractiveSession()
    else:
        _SESSION = tf.get_default_session()

    return _SESSION


def gmm_ms(X):
    aics = []
    n_components_range = range(1, 20)
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GMM(n_components=n_components,
                          covariance_type="full")
        gmm.fit(X)
        aics.append(gmm.aic(X))
    return np.array(aics)

def analyze_div(X_real, X_sample):
    
    def kl_div(p, q):
        eps = 1e-10
        p_safe = np.copy(p)
        p_safe[p_safe < eps] = eps
        q_safe = np.copy(q)
        q_safe[q_safe < eps] = eps
        return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))

    def js_div(p, q):
        m = (p + q) / 2.
        return (kl_div(p, m) + kl_div(q, m))/2.
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_trans_real = pca.fit_transform(X_real)
    X_trans_fake = pca.transform(X_sample)
    
    from scipy import stats

    def cartesian_prod(x, y):
        return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

    dx = 0.1
    dy = 0.1
    
    xmin1 = np.min(X_trans_real[:, 0]) - 3.0
    xmax1 = np.max(X_trans_real[:, 0]) + 3.0
    
    xmin2 = np.min(X_trans_real[:, 1]) - 3.0
    xmax2 = np.max(X_trans_real[:, 1]) + 3.0
    
    space = cartesian_prod(np.arange(xmin1,xmax1,dx), np.arange(xmin2,xmax2,dy)).T

    real_kde = stats.gaussian_kde(X_trans_real.T)
    real_density = real_kde(space) * dx * dy

    fake_kde = stats.gaussian_kde(X_trans_fake.T)
    fake_density = fake_kde(space) * dx * dy
    

    return js_div(real_density, fake_density), X_trans_real, X_trans_fake
    


def bgan_synth(synth_dataset, z_dim, batch_size=64, numz=5, num_iter=1000, 
               wasserstein=False, rpath="synth_results",
               base_learning_rate=1e-2, lr_decay=3., save_weights=False):

    bgan = BGAN([synth_dataset.x_dim], z_dim,
                synth_dataset.N,
                batch_size=batch_size,
                prior_std=10.0, alpha=1e-3,
                J=numz, M=1, ml=(numz == 1),
                num_classes=1,
                wasserstein=wasserstein, # unsupervised only
                )
    print "Starting session"
    session = get_session()
    tf.global_variables_initializer().run()

    print "Starting training loop"
        
    num_train_iter = num_iter

    sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
    
    all_aics_fake, all_data_fake, all_dists = [], [], []

    for train_iter in range(num_train_iter):

        learning_rate = base_learning_rate * np.exp(-lr_decay *
                                                    min(1.0, (train_iter*batch_size)/float(synth_dataset.N)))
        print learning_rate
        
        batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
        
        input_batch = synth_dataset.next_batch(batch_size)
        _, d_loss = session.run([bgan.d_optim, bgan.d_loss], feed_dict={bgan.inputs: input_batch,
                                                                        bgan.z: batch_z,
                                                                        bgan.d_learning_rate: learning_rate})
        if wasserstein:
            session.run(bgan.clip_d, feed_dict={})

        g_losses = []
        for gi in xrange(bgan.num_gen):

            # compute g_sample loss
            batch_z = np.random.uniform(-1, 1, [batch_size, z_dim])
            _, g_loss = session.run([bgan.g_optims[gi], bgan.generation["g_losses"][gi]],
                                    feed_dict={bgan.z: batch_z,
                                               bgan.g_learning_rate: learning_rate})
            g_losses.append(g_loss)

        print "Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % gl for gl in g_losses]))

        if (train_iter + 1) % 100 == 0:
            print "Disc loss = %.2f, Gen loss = %s" % (d_loss, ", ".join(["%.2f" % gl for gl in g_losses]))
            print "Running GMM on sampled data"
            fake_data = []
            for num_samples in xrange(10):
                for gi in xrange(bgan.num_gen):
                    # collect sample
                    sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                    sampled_data = session.run(bgan.generation["gen_samplers"][gi], feed_dict={bgan.z: sample_z})
                    fake_data.append(sampled_data)

            X_real = synth_dataset.X
            X_sample = np.concatenate(fake_data)

            all_data_fake.append(X_sample)

            aics_fake = gmm_ms(X_sample)
            print "Fake number of clusters (AIC estimate):", aics_fake.argmin()
            dist, X_trans_real, X_trans_fake = analyze_div(X_real, X_sample)
            print "JS div:", dist
            fp = FigPrinter((1,2))
            xmin1 = np.min(X_trans_real[:, 0]) - 1.0
            xmax1 = np.max(X_trans_real[:, 0]) + 1.0
            xmin2 = np.min(X_trans_real[:, 1]) - 1.0
            xmax2 = np.max(X_trans_real[:, 1]) + 1.0
            fp.ax_arr[0].plot(X_trans_real[:, 0], X_trans_real[:, 1], '.r')
            fp.ax_arr[0].set_xlim([xmin1, xmax1]); fp.ax_arr[0].set_ylim([xmin2, xmax2])
            fp.ax_arr[1].plot(X_trans_fake[:, 0], X_trans_fake[:, 1], '.g')
            fp.ax_arr[1].set_xlim([xmin1, xmax1]); fp.ax_arr[1].set_ylim([xmin2, xmax2])
            fp.ax_arr[0].set_aspect('equal', adjustable='box')
            fp.ax_arr[1].set_aspect('equal', adjustable='box')
            fp.ax_arr[1].set_title("Iter %i" % (train_iter+1))            
            fp.print_to_file(os.path.join(rpath, "pca_distribution_%i_%i.png" % (numz, train_iter+1)))

            all_aics_fake.append(aics_fake)
            all_dists.append(dist)

            if save_weights:
                var_dict = {}
                for var in tf.trainable_variables():
                    var_dict[var.name] = session.run(var.name)

                np.savez_compressed(os.path.join(rpath,
                                                 "weights_%i.npz" % train_iter),
                                    **var_dict)

            
            

            
    return {"data_fake": all_data_fake,
            "data_real": synth_dataset.X,
            "z_dim": z_dim,
            "numz": numz,
            "num_iter": num_iter,
            "divergences": all_dists,
            "all_aics_fake": np.array(all_aics_fake)}


if __name__ == "__main__":

    import argparse
    import time

    parser = argparse.ArgumentParser(description='Script to run Bayesian GAN synthetic experiments')

    parser.add_argument('--x_dim',
                        type=int,
                        default=100,
                        help='dim of x for synthetic data')
    parser.add_argument('--z_dim',
                        type=int,
                        default=2,
                        help='dim of z for generator')
    parser.add_argument('--train_iter',
                        type=int,
                        default=1000,
                        help='no of GAN iterations')
    parser.add_argument('--numz',
                        type=int,
                        default=1,
                        help='no of z samples')
    parser.add_argument('--wasserstein',
                        action="store_true",
                        help='use wasserstein GAN')
    parser.add_argument('--out_dir',
                        default="/tmp/synth_results",
                        help='path of where to store results')
    parser.add_argument('--random_seed',
                        type=int,
                        default=2222,
                        help='set seed for repeatability')
    parser.add_argument('--save_weights',
                        action="store_true",
                        help='whether to save weight vectors')

    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)
        tf.set_random_seed(args.random_seed)

    if not os.path.exists(args.out_dir):
        print "Creating %s" % args.out_dir
        os.makedirs(args.out_dir)

    results_path = os.path.join(args.out_dir, "experiment_%i" % (int(time.time())))
    os.makedirs(results_path)
    import pprint
    with open(os.path.join(results_path, "args.txt"), "w") as hf:
        hf.write("Experiment settings:\n")
        hf.write("%s\n" % (pprint.pformat(args.__dict__)))
    
    synth_d = SynthDataset(args.x_dim)

    results = bgan_synth(synth_d, args.z_dim, num_iter=args.train_iter, 
                         numz=args.numz, wasserstein=args.wasserstein, 
                         rpath=results_path, save_weights=args.save_weights)

    np.savez(os.path.join(results_path, "run_%s_%s.npz" % ("wasserstein" if args.wasserstein else "regular",
                                                           "ml" if args.numz == 1 else "bayes")),
             **results)




