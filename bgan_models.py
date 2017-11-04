import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict


#### Bayesian DCGAN

from dcgan_ops import *

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class BGAN(object):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, prior_std=1.0, J=1, M=1, 
                 num_classes=1, alpha=0.01, lr=0.0002, gen_observed=1000,
                 optimizer='adam', wasserstein=False, ml=False):

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.gen_observed = gen_observed
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.optimizer = optimizer.lower()

        self.wasserstein = wasserstein
        if self.wasserstein:
            assert num_classes == 1, "cannot do semi-sup learning with wasserstein ... yet"
            
        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_mcmc = M
        self.alpha = alpha
        self.lr = lr

        self.ml = ml
        if self.ml:
            assert self.num_gen*self.num_mcmc == 1, "cannot have multiple generators in ml mode"
        
        self.weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, 1000)),
                                        ("g_h0_lin_b", (1000,)),
                                        ("g_lin_W", (1000, self.x_dim[0])),
                                        ("g_lin_b", (self.x_dim[0],))])
        
        self.sghmc_noise = {}
        self.noise_std = np.sqrt(2 * self.alpha)
        for name, dim in self.weight_dims.iteritems():
            self.sghmc_noise[name] = tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(self.weight_dims[name]))

        self.K = num_classes # 1 means unsupervised, label == 0 always reserved for fake

        self.build_bgan_graph()

        if self.K > 1:
            self.build_test_graph()

    def _get_optimizer(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")

    def build_test_graph(self):

        self.test_inputs = tf.placeholder(tf.float32,
                                          [self.batch_size] + self.x_dim, name='real_test_images')
        
        self.lbls = tf.placeholder(tf.float32,
                                   [self.batch_size, self.K], name='real_sup_targets')

        self.S, self.S_logits = self.sup_discriminator(self.inputs, self.K)

        self.test_D, self.test_D_logits = self.discriminator(self.test_inputs, self.K+1, reuse=True)
        self.test_S, self.test_S_logits = self.sup_discriminator(self.test_inputs, self.K, reuse=True)

        self.s_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.S_logits,
                                                                             labels=self.lbls))

        t_vars = tf.trainable_variables()
        self.sup_vars = [var for var in t_vars if 'sup_' in var.name]

        # this is purely supervised
        supervised_lr = 0.05 * self.lr
        s_opt = self._get_optimizer(supervised_lr)
        self.s_optim = s_opt.minimize(self.s_loss, var_list=self.sup_vars)
        s_opt_adam = tf.train.AdamOptimizer(learning_rate=supervised_lr, beta1=0.5)
        self.s_optim_adam = s_opt_adam.minimize(self.s_loss, var_list=self.sup_vars)


    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size] + self.x_dim, name='real_images')
        
        self.labeled_inputs = tf.placeholder(tf.float32,
                                             [self.batch_size] + self.x_dim, name='real_images_w_labels')
        
        self.labels = tf.placeholder(tf.float32,
                                     [self.batch_size, self.K+1], name='real_targets')


        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        #self.z_sum = histogram_summary("z", self.z) TODO looks cool

        self.gen_param_list = []
        with tf.variable_scope("generator") as scope:
            for gi in xrange(self.num_gen):
                for m in xrange(self.num_mcmc):
                    gen_params = AttributeDict()
                    for name, shape in self.weight_dims.iteritems():
                        gen_params[name] = tf.get_variable("%s_%04d_%04d" % (name, gi, m),
                                                           shape, initializer=tf.random_normal_initializer(stddev=0.02))
                    self.gen_param_list.append(gen_params)

        self.D, self.D_logits = self.discriminator(self.inputs, self.K+1)
        self.Dsup, self.Dsup_logits = self.discriminator(self.labeled_inputs, self.K+1, reuse=True)

        if self.K == 1:
            if self.wasserstein:
                self.d_loss_real = tf.reduce_mean(self.D_logits)
            else:
                # regular GAN
                constant_labels = np.zeros((self.batch_size, 2))
                constant_labels[:, 1] = 1.0
                self.d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.D_logits,
                                                                                          labels=tf.constant(constant_labels)))
        else:
            self.d_loss_sup = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.Dsup_logits,
                                                                                     labels=self.labels))    
            self.d_loss_real = -tf.reduce_mean(tf.log((1.0 - self.D[:, 0]) + 1e-8))
                        

        self.generation = defaultdict(list)
        for gen_params in self.gen_param_list:
            self.generation["g_prior"].append(self.gen_prior(gen_params))
            self.generation["g_noise"].append(self.gen_noise(gen_params))
            self.generation["generators"].append(self.generator(self.z, gen_params))
            self.generation["gen_samplers"].append(self.sampler(self.z, gen_params))
            D_, D_logits_ = self.discriminator(self.generator(self.z, gen_params), self.K+1, reuse=True)
            self.generation["d_logits"].append(D_logits_)
            self.generation["d_probs"].append(D_)
            

        all_d_logits = tf.concat(self.generation["d_logits"], 0)
        if self.wasserstein:
            self.d_loss_fake = -tf.reduce_mean(all_d_logits)
        else:
            constant_labels = np.zeros((self.batch_size*self.num_gen*self.num_mcmc, self.K+1))
            constant_labels[:, 0] = 1.0 # class label indicating it came from generator, aka fake
            self.d_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=all_d_logits,
                                                                                      labels=tf.constant(constant_labels)))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

        self.d_loss = self.d_loss_real + self.d_loss_fake
        if not self.ml:
            self.d_loss += self.disc_prior() + self.disc_noise()
        if self.K > 1:
            self.d_loss_semi = self.d_loss_sup + self.d_loss_real + self.d_loss_fake
            if not self.ml:
                self.d_loss_semi += self.disc_prior() + self.disc_noise()

        self.g_vars = []
        for gi in xrange(self.num_gen):
            for m in xrange(self.num_mcmc):
                self.g_vars.append([var for var in t_vars if 'g_' in var.name and "_%04d_%04d" % (gi, m) in var.name])

        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])
        d_opt = self._get_optimizer(self.d_learning_rate)
        self.d_optim = d_opt.minimize(self.d_loss, var_list=self.d_vars)

        d_opt_adam = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=0.5)
        self.d_optim_adam = d_opt_adam.minimize(self.d_loss, var_list=self.d_vars)
        
        clip_d = [w.assign(tf.clip_by_value(w, -0.01, 0.01)) for w in self.d_vars]
        self.clip_d = clip_d

        if self.K > 1:
            self.d_semi_learning_rate = tf.placeholder(tf.float32, shape=[])
            d_opt_semi = self._get_optimizer(self.d_semi_learning_rate)
            self.d_optim_semi = d_opt_semi.minimize(self.d_loss_semi, var_list=self.d_vars)
            d_opt_semi_adam = tf.train.AdamOptimizer(learning_rate=self.d_semi_learning_rate, beta1=0.5)
            self.d_optim_semi_adam = d_opt_semi_adam.minimize(self.d_loss_semi, var_list=self.d_vars)
        
        self.g_optims, self.g_optims_adam = [], []
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])
        for gi in xrange(self.num_gen*self.num_mcmc):
            if self.wasserstein:
                g_loss = tf.reduce_mean(self.generation["d_logits"][gi])
            else:
                g_loss = -tf.reduce_mean(tf.log((1.0 - self.generation["d_probs"][gi][:, 0]) + 1e-8))
            if not self.ml:
                g_loss += self.generation["g_prior"][gi] + self.generation["g_noise"][gi]
            self.generation["g_losses"].append(g_loss)
            g_opt = self._get_optimizer(self.g_learning_rate)
            self.g_optims.append(g_opt.minimize(g_loss, var_list=self.g_vars[gi]))
            g_opt_adam = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
            self.g_optims_adam.append(g_opt_adam.minimize(g_loss, var_list=self.g_vars[gi]))

            
    def discriminator(self, x, K, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(linear(x, 1000, 'd_lin_0'))
            h1 = linear(h0, K, 'd_lin_1')
            return tf.nn.softmax(h1), h1

    def sup_discriminator(self, x, K, reuse=False):
        
        pass
        
    def generator(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            h0 = lrelu(linear(z, 1000, 'g_h0_lin',
                              matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b))
            self.x_ = linear(h0, self.x_dim[0], 'g_lin',
                             matrix=gen_params.g_lin_W, bias=gen_params.g_lin_b)
            return self.x_

    def sampler(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            return self.generator(z, gen_params)

    def gen_prior(self, gen_params):
        with tf.variable_scope("generator") as scope:
            prior_loss = 0.0
            for var in gen_params.values():
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.gen_observed

        return prior_loss

    def gen_noise(self, gen_params): # for SGHMC
        with tf.variable_scope("generator") as scope:
            noise_loss = 0.0
            for name, var in gen_params.iteritems():
                noise_loss += tf.reduce_sum(var * self.sghmc_noise[name].sample())
        noise_loss /= self.gen_observed
        return noise_loss

    def disc_prior(self):
        with tf.variable_scope("discriminator") as scope:
            prior_loss = 0.0
            for var in self.d_vars:
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.dataset_size

        return prior_loss

    def disc_noise(self): # for SGHMC
        with tf.variable_scope("discriminator") as scope:
            noise_loss = 0.0
            for var in self.d_vars:
                noise_ = tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())
        noise_loss /= self.dataset_size
        return noise_loss


class BDCGAN(BGAN):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, gf_dim=64, df_dim=64, 
                 prior_std=1.0, J=1, M=1, num_classes=1, eta=2e-4, 
                 alpha=0.01, lr=0.0002, optimizer='adam', wasserstein=False, 
                 ml=False, gen_observed=1000):

        assert len(x_dim) == 3, "invalid image dims"
        
        c_dim = x_dim[2]
        self.is_grayscale = (c_dim == 1)
        self.optimizer = optimizer.lower()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.gen_observed = gen_observed

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.lr = lr

        self.wasserstein = wasserstein
        
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.sd_bn1 = batch_norm(name='sd_bn1')
        self.sd_bn2 = batch_norm(name='sd_bn2')
        self.sd_bn3 = batch_norm(name='sd_bn3')


        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')

        self.wasserstein = wasserstein
        
        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_mcmc = M
        self.eta = eta
        self.alpha = alpha
        # ML
        self.ml = ml
        if self.ml:
            assert self.num_gen == 1, "cannot have >1 generator for ml"

        self.output_height = x_dim[0]
        self.output_width = x_dim[1]

        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
        
        self.gen_params = AttributeDict()
        self.bgen_params = AttributeDict()
        self.weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, self.gf_dim * 8 * s_h16 * s_w16)),
                                        ("g_h0_lin_b", (self.gf_dim * 8 * s_h16 * s_w16,)),
                                        ("g_h1_W", (5, 5, self.gf_dim*4, self.gf_dim*8)),
                                        ("g_h1_b", (self.gf_dim*4,)),
                                        ("g_h2_W", (5, 5, self.gf_dim*2, self.gf_dim*4)),
                                        ("g_h2_b", (self.gf_dim*2,)),
                                        ("g_h3_W", (5, 5, self.gf_dim*1, self.gf_dim*2)),
                                        ("g_h3_b", (self.gf_dim*1,)),
                                        ("g_h4_W", (5, 5, self.c_dim, self.gf_dim*1)),
                                        ("g_h4_b", (self.c_dim,))])

        self.sghmc_noise = {}
        self.noise_std = np.sqrt(2 * self.alpha * self.eta)
        for name, dim in self.weight_dims.iteritems():
            self.sghmc_noise[name] = tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(self.weight_dims[name]))

        self.K = num_classes # 1 means unsupervised, label == 0 always reserved for fake

        self.build_bgan_graph()

        if self.K > 1:
            self.build_test_graph()
                                             
                    
    def discriminator(self, image, K, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0,
                                         self.df_dim * 2,
                                         name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1,
                                         self.df_dim * 4,
                                         name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2,
                                         self.df_dim * 8,
                                         name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), K, 'd_h3_lin')
            return tf.nn.softmax(h4), h4
        

    def sup_discriminator(self, image, K, reuse=False):
        with tf.variable_scope("sup_discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='sup_h0_conv'))
            h1 = lrelu(self.sd_bn1(conv2d(h0,
                                         self.df_dim * 2,
                                         name='sup_h1_conv')))
            h2 = lrelu(self.sd_bn2(conv2d(h1,
                                         self.df_dim * 4,
                                         name='sup_h2_conv')))
            h3 = lrelu(self.sd_bn3(conv2d(h2,
                                         self.df_dim * 8,
                                         name='sup_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), K, 'sup_h3_lin')
            return tf.nn.softmax(h4), h4

            

    def generator(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True,
                                                   matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b)

            self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0,
                                                     [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True,
                                                     w=gen_params.g_h1_W, biases=gen_params.g_h1_b)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                                                [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True,
                                                w=gen_params.g_h2_W, biases=gen_params.g_h2_b)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                                                [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True,
                                                w=gen_params.g_h3_W, biases=gen_params.g_h3_b)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                                                [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True,
                                                w=gen_params.g_h4_W, biases=gen_params.g_h4_b)

            return tf.nn.tanh(h4)
        

    def sampler(self, z, gen_params):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            z_ = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin',
                        matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b)

            h0 = tf.reshape(z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0,
                          [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1',
                          w=gen_params.g_h1_W, biases=gen_params.g_h1_b)
            
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1,
                          [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2',
                          w=gen_params.g_h2_W, biases=gen_params.g_h2_b)
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2,
                          [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3',
                          w=gen_params.g_h3_W, biases=gen_params.g_h3_b)
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3,
                          [self.batch_size, s_h, s_w, self.c_dim], name='g_h4',
                          w=gen_params.g_h4_W, biases=gen_params.g_h4_b)

            return tf.nn.tanh(h4)


