import numpy as np
import tensorflow as tf

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict

from dcgan_ops import *


def conv_out_size(size, stride):
    co = int(math.ceil(size / float(stride)))
    return co

def kernel_sizer(size, stride):
    ko = int(math.ceil(size / float(stride)))
    if ko % 2 == 0:
        ko += 1
    return ko


class BDCGAN(object):

    def __init__(self, x_dim, z_dim, dataset_size, batch_size=64, gf_dim=64, df_dim=64, 
                 prior_std=1.0, J=1, M=1, eta=2e-4, num_layers=4,
                 alpha=0.01, lr=0.0002, optimizer='adam', wasserstein=False, 
                 ml=False, J_d=None):


        assert len(x_dim) == 3, "invalid image dims"
        c_dim = x_dim[2]
        self.is_grayscale = (c_dim == 1)
        self.optimizer = optimizer.lower()
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        
        self.K = 2 # fake and real classes
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.lr = lr
        
        # Bayes
        self.prior_std = prior_std
        self.num_gen = J
        self.num_disc = J_d if J_d is not None else 1
        self.num_mcmc = M
        self.eta = eta
        self.alpha = alpha
        # ML
        self.ml = ml
        if self.ml:
            assert self.num_gen == 1 and self.num_disc == 1 and self.num_mcmc == 1, "invalid settings for ML training"

        self.noise_std = np.sqrt(2 * self.alpha * self.eta)

        def get_strides(num_layers, num_pool):
            interval = int(math.floor(num_layers/float(num_pool)))
            strides = np.array([1]*num_layers)
            strides[0:interval*num_pool:interval] = 2
            return strides

        self.num_pool = 4
        self.max_num_dfs = 512
        self.gen_strides = get_strides(num_layers, self.num_pool)
        self.disc_strides = self.gen_strides
        num_dfs = np.cumprod(np.array([self.df_dim] + list(self.disc_strides)))[:-1]
        num_dfs[num_dfs >= self.max_num_dfs] = self.max_num_dfs # memory
        self.num_dfs = list(num_dfs)
        self.num_gfs = self.num_dfs[::-1]

        self.construct_from_hypers(gen_strides=self.gen_strides, disc_strides=self.disc_strides,
                                   num_gfs=self.num_gfs, num_dfs=self.num_dfs)
        
        self.build_bgan_graph()
        

    def construct_from_hypers(self, gen_kernel_size=5, gen_strides=[2,2,2,2],
                              disc_kernel_size=5, disc_strides=[2,2,2,2],
                              num_dfs=None, num_gfs=None):

        
        self.d_batch_norm = AttributeDict([("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) for dbn_i in range(len(disc_strides))])
        self.sup_d_batch_norm = AttributeDict([("sd_bn%i" % dbn_i, batch_norm(name='sup_d_bn%i' % dbn_i)) for dbn_i in range(5)])
        self.g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(len(gen_strides))])

        if num_dfs is None:
            num_dfs = [self.df_dim, self.df_dim*2, self.df_dim*4, self.df_dim*8]
            
        if num_gfs is None:
            num_gfs = [self.gf_dim*8, self.gf_dim*4, self.gf_dim*2, self.gf_dim]

        assert len(gen_strides) == len(num_gfs), "invalid hypers!"
        assert len(disc_strides) == len(num_dfs), "invalid hypers!"

        s_h, s_w = self.x_dim[0], self.x_dim[1]
        ks = gen_kernel_size
        self.gen_output_dims = OrderedDict()
        self.gen_weight_dims = OrderedDict()
        num_gfs = num_gfs + [self.c_dim]
        self.gen_kernel_sizes = [ks]
        for layer in range(len(gen_strides))[::-1]:
            self.gen_output_dims["g_h%i_out" % (layer+1)] = (s_h, s_w)
            assert gen_strides[layer] <= 2, "invalid stride"
            assert ks % 2 == 1, "invalid kernel size"
            self.gen_weight_dims["g_h%i_W" % (layer+1)] = (ks, ks, num_gfs[layer+1], num_gfs[layer])
            self.gen_weight_dims["g_h%i_b" % (layer+1)] = (num_gfs[layer+1],)
            s_h, s_w = conv_out_size(s_h, gen_strides[layer]), conv_out_size(s_w, gen_strides[layer])
            ks = kernel_sizer(ks, gen_strides[layer])
            self.gen_kernel_sizes.append(ks)


        self.gen_weight_dims.update(OrderedDict([("g_h0_lin_W", (self.z_dim, num_gfs[0] * s_h * s_w)),
                                                 ("g_h0_lin_b", (num_gfs[0] * s_h * s_w,))]))
        self.gen_output_dims["g_h0_out"] = (s_h, s_w)

        self.disc_weight_dims = OrderedDict()
        s_h, s_w = self.x_dim[0], self.x_dim[1]
        num_dfs = [self.c_dim] + num_dfs
        ks = disc_kernel_size
        self.disc_kernel_sizes = [ks]
        for layer in range(len(disc_strides)):
            assert disc_strides[layer] <= 2, "invalid stride"
            assert ks % 2 == 1, "invalid kernel size"
            self.disc_weight_dims["d_h%i_W" % layer] = (ks, ks, num_dfs[layer], num_dfs[layer+1])
            self.disc_weight_dims["d_h%i_b" % layer] = (num_dfs[layer+1],)
            s_h, s_w = conv_out_size(s_h, disc_strides[layer]), conv_out_size(s_w, disc_strides[layer])
            ks = kernel_sizer(ks, disc_strides[layer])
            self.disc_kernel_sizes.append(ks)

        self.disc_weight_dims.update(OrderedDict([("d_h_end_lin_W", (num_dfs[-1] * s_h * s_w, num_dfs[-1])),
                                                  ("d_h_end_lin_b", (num_dfs[-1],)),
                                                  ("d_h_out_lin_W", (num_dfs[-1], self.K)),
                                                  ("d_h_out_lin_b", (self.K,))]))


        for k, v in self.gen_output_dims.items():
            print "%s: %s" % (k, v)
        print '****'
        for k, v in self.gen_weight_dims.items():
            print "%s: %s" % (k, v)
        print '****'
        for k, v in self.disc_weight_dims.items():
            print "%s: %s" % (k, v)





    def construct_nets(self):

        self.num_disc_layers = 5
        self.num_gen_layers = 5
        self.d_batch_norm = AttributeDict([("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) for dbn_i in range(self.num_disc_layers)])
        self.sup_d_batch_norm = AttributeDict([("sd_bn%i" % dbn_i, batch_norm(name='sup_d_bn%i' % dbn_i)) for dbn_i in range(self.num_disc_layers)])
        self.g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(self.num_gen_layers)])


        s_h, s_w = self.x_dim[0], self.x_dim[1]
        s_h2, s_w2 = conv_out_size(s_h, 2), conv_out_size(s_w, 2)
        s_h4, s_w4 = conv_out_size(s_h2, 2), conv_out_size(s_w2, 2)
        s_h8, s_w8 = conv_out_size(s_h4, 2), conv_out_size(s_w4, 2)
        s_h16, s_w16 = conv_out_size(s_h8, 2), conv_out_size(s_w8, 2)

        self.gen_output_dims = OrderedDict([("g_h0_out", (s_h16, s_w16)),
                                            ("g_h1_out", (s_h8, s_w8)),
                                            ("g_h2_out", (s_h4, s_w4)),
                                            ("g_h3_out", (s_h2, s_w2)),
                                            ("g_h4_out", (s_h, s_w))])

        
        self.gen_weight_dims = OrderedDict([("g_h0_lin_W", (self.z_dim, self.gf_dim * 8 * s_h16 * s_w16)),
                                            ("g_h0_lin_b", (self.gf_dim * 8 * s_h16 * s_w16,)),
                                            ("g_h1_W", (5, 5, self.gf_dim*4, self.gf_dim*8)),
                                            ("g_h1_b", (self.gf_dim*4,)),
                                            ("g_h2_W", (5, 5, self.gf_dim*2, self.gf_dim*4)),
                                            ("g_h2_b", (self.gf_dim*2,)),
                                            ("g_h3_W", (5, 5, self.gf_dim*1, self.gf_dim*2)),
                                            ("g_h3_b", (self.gf_dim*1,)),
                                            ("g_h4_W", (5, 5, self.c_dim, self.gf_dim*1)),
                                            ("g_h4_b", (self.c_dim,))])

        self.disc_weight_dims = OrderedDict([("d_h0_W", (5, 5, self.c_dim, self.df_dim)),
                                             ("d_h0_b", (self.df_dim,)),
                                             ("d_h1_W", (5, 5, self.df_dim, self.df_dim*2)),
                                             ("d_h1_b", (self.df_dim*2,)),
                                             ("d_h2_W", (5, 5, self.df_dim*2, self.df_dim*4)),
                                             ("d_h2_b", (self.df_dim*4,)),
                                             ("d_h3_W", (5, 5, self.df_dim*4, self.df_dim*8)),
                                             ("d_h3_b", (self.df_dim*8,)),
                                             ("d_h_end_lin_W", (self.df_dim * 8 * s_h16 * s_w16, self.df_dim*4)),
                                             ("d_h_end_lin_b", (self.df_dim*4,)),
                                             ("d_h_out_lin_W", (self.df_dim*4, self.K)),
                                             ("d_h_out_lin_b", (self.K,))])



    def _get_optimizer(self, lr):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        elif self.optimizer == 'sgd':
            return tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.5)
        else:
            raise ValueError("Optimizer must be either 'adam' or 'sgd'")    

    def initialize_wgts(self, scope_str):

        if scope_str == "generator":
            weight_dims = self.gen_weight_dims
            numz = self.num_gen
        elif scope_str == "discriminator":
            weight_dims = self.disc_weight_dims
            numz = self.num_disc
        else:
            raise RuntimeError("invalid scope!")

        param_list = []
        with tf.variable_scope(scope_str) as scope:
            for zi in xrange(numz):
                for m in xrange(self.num_mcmc):
                    wgts_ = AttributeDict()
                    for name, shape in weight_dims.iteritems():
                        wgts_[name] = tf.get_variable("%s_%04d_%04d" % (name, zi, m),
                                                      shape, initializer=tf.random_normal_initializer(stddev=0.02))
                    param_list.append(wgts_)
            return param_list
        

    def build_bgan_graph(self):
    
        self.inputs = tf.placeholder(tf.float32,
                                     [self.batch_size] + self.x_dim, name='real_images')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim, self.num_gen], name='z')
        self.z_sampler = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z_sampler')
        
        # initialize generator weights
        self.gen_param_list = self.initialize_wgts("generator")
        self.disc_param_list = self.initialize_wgts("discriminator")
        ### build discrimitive losses and optimizers
        # prep optimizer args
        self.d_learning_rate = tf.placeholder(tf.float32, shape=[])
        
        # compile all disciminative weights
        t_vars = tf.trainable_variables()
        self.d_vars = []
        for di in xrange(self.num_disc):
            for m in xrange(self.num_mcmc):
                self.d_vars.append([var for var in t_vars if 'd_' in var.name and "_%04d_%04d" % (di, m) in var.name])

        ### build disc losses and optimizers
        self.d_losses, self.d_optims, self.d_optims_adam = [], [], []
        for di, disc_params in enumerate(self.disc_param_list):

            d_probs, d_logits, _ = self.discriminator(self.inputs, self.K, disc_params)

            constant_labels = np.zeros((self.batch_size, 2))
            constant_labels[:, 1] = 1.0  # real
            d_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits,
                                                                                 labels=tf.constant(constant_labels)))

            d_loss_fakes = []
            for gi, gen_params in enumerate(self.gen_param_list):
                d_probs_, d_logits_, _ = self.discriminator(self.generator(self.z[:, :, gi % self.num_gen], gen_params), 
                                                            self.K, disc_params)
                constant_labels = np.zeros((self.batch_size, self.K))
                constant_labels[:, 0] = 1.0 # class label indicating it came from generator, aka fake
                d_loss_fake_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_,
                                                                                      labels=tf.constant(constant_labels)))
                d_loss_fakes.append(d_loss_fake_)

            d_losses = []
            for d_loss_fake_ in d_loss_fakes:
                d_loss_ = d_loss_real * float(self.num_gen) + d_loss_fake_
                if not self.ml:
                    d_loss_ += self.disc_prior(disc_params) + self.disc_noise(disc_params)
                d_losses.append(tf.reshape(d_loss_, [1]))

            d_loss = tf.reduce_logsumexp(tf.concat(d_losses, 0))
            self.d_losses.append(d_loss)
            d_opt = self._get_optimizer(self.d_learning_rate)
            self.d_optims.append(d_opt.minimize(d_loss, var_list=self.d_vars[di]))
            d_opt_adam = tf.train.AdamOptimizer(learning_rate=self.d_learning_rate, beta1=0.5)
            self.d_optims_adam.append(d_opt_adam.minimize(d_loss, var_list=self.d_vars[di]))

        ### build generative losses and optimizers
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.g_vars = []
        for gi in xrange(self.num_gen):
            for m in xrange(self.num_mcmc):
                self.g_vars.append([var for var in t_vars if 'g_' in var.name and "_%04d_%04d" % (gi, m) in var.name])
        
        self.g_losses, self.g_optims, self.g_optims_adam = [], [], []
        for gi, gen_params in enumerate(self.gen_param_list):

            gi_losses = []
            for disc_params in self.disc_param_list:
                d_probs_, d_logits_, d_features_fake = self.discriminator(self.generator(self.z[:, :, gi % self.num_gen],
                                                                                         gen_params),
                                                                          self.K, disc_params)
                _, _, d_features_real = self.discriminator(self.inputs, self.K, disc_params)
                constant_labels = np.zeros((self.batch_size, self.K))
                constant_labels[:, 1] = 1.0 # class label indicating that this fake is real
                g_loss_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d_logits_,
                                                                                 labels=tf.constant(constant_labels)))
                g_loss_ += tf.reduce_mean(huber_loss(d_features_real[-1], d_features_fake[-1]))
                if not self.ml:
                    g_loss_ += self.gen_prior(gen_params) + self.gen_noise(gen_params)
                gi_losses.append(tf.reshape(g_loss_, [1]))
                
            g_loss = tf.reduce_logsumexp(tf.concat(gi_losses, 0))
            self.g_losses.append(g_loss)
            g_opt = self._get_optimizer(self.g_learning_rate)
            self.g_optims.append(g_opt.minimize(g_loss, var_list=self.g_vars[gi]))
            g_opt_adam = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
            self.g_optims_adam.append(g_opt_adam.minimize(g_loss, var_list=self.g_vars[gi]))

        ### build samplers
        self.gen_samplers = []
        for gi, gen_params in enumerate(self.gen_param_list):
            self.gen_samplers.append(self.generator(self.z_sampler, gen_params))


    def discriminator(self, image, K, disc_params, train=True):

        with tf.variable_scope("discriminator") as scope:

            h = image
            for layer in range(len(self.disc_strides)):
                if layer == 0:
                    h = lrelu(conv2d(h,
                                     self.disc_weight_dims["d_h%i_W" % layer][-1],
                                     name='d_h%i_conv' % layer,
                                     k_h=self.disc_kernel_sizes[layer], k_w=self.disc_kernel_sizes[layer],
                                     d_h=self.disc_strides[layer], d_w=self.disc_strides[layer],
                                     w=disc_params["d_h%i_W" % layer], biases=disc_params["d_h%i_b" % layer]))
                else:
                    h = lrelu(self.d_batch_norm["d_bn%i" % layer](conv2d(h,
                                                                         self.disc_weight_dims["d_h%i_W" % layer][-1],
                                                                         name='d_h%i_conv' % layer,
                                                                         k_h=self.disc_kernel_sizes[layer], k_w=self.disc_kernel_sizes[layer],
                                                                         d_h=self.disc_strides[layer], d_w=self.disc_strides[layer],
                                                                         w=disc_params["d_h%i_W" % layer], biases=disc_params["d_h%i_b" % layer]), train=train))

            h_end = lrelu(linear(tf.reshape(h, [self.batch_size, -1]),
                              self.df_dim*4, "d_h_end_lin",
                              matrix=disc_params.d_h_end_lin_W, bias=disc_params.d_h_end_lin_b)) # for feature norm
            h_out = linear(h_end, K,
                           'd_h_out_lin',
                           matrix=disc_params.d_h_out_lin_W, bias=disc_params.d_h_out_lin_b)
            
            return tf.nn.softmax(h_out), h_out, [h_end]
            

    def generator(self, z, gen_params):

        with tf.variable_scope("generator") as scope:

            h = linear(z, self.gen_weight_dims["g_h0_lin_W"][-1], 'g_h0_lin',
                       matrix=gen_params.g_h0_lin_W, bias=gen_params.g_h0_lin_b)
            h = tf.nn.relu(self.g_batch_norm.g_bn0(h))

            h = tf.reshape(h, [self.batch_size, self.gen_output_dims["g_h0_out"][0],
                               self.gen_output_dims["g_h0_out"][1], -1])

            for layer in range(1, len(self.gen_strides)+1):

                out_shape = [self.batch_size, self.gen_output_dims["g_h%i_out" % layer][0],
                             self.gen_output_dims["g_h%i_out" % layer][1], self.gen_weight_dims["g_h%i_W" % layer][-2]]

                h = deconv2d(h,
                             out_shape,
                             k_h=self.gen_kernel_sizes[layer-1], k_w=self.gen_kernel_sizes[layer-1],
                             d_h=self.gen_strides[layer-1], d_w=self.gen_strides[layer-1],
                             name='g_h%i' % layer,
                             w=gen_params["g_h%i_W" % layer], biases=gen_params["g_h%i_b" % layer])
                if layer < len(self.gen_strides):
                    h = tf.nn.relu(self.g_batch_norm["g_bn%i" % layer](h))

            return tf.nn.tanh(h)        


    def gen_prior(self, gen_params):
        with tf.variable_scope("generator") as scope:
            prior_loss = 0.0
            for var in gen_params.values():
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.dataset_size

        return prior_loss

    def gen_noise(self, gen_params): 
        with tf.variable_scope("generator") as scope:
            noise_loss = 0.0
            for name, var in gen_params.iteritems():
                noise_ = tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())
        noise_loss /= self.dataset_size
        return noise_loss

    def disc_prior(self, disc_params):
        with tf.variable_scope("discriminator") as scope:
            prior_loss = 0.0
            for var in disc_params.values():
                nn = tf.divide(var, self.prior_std)
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))
                
        prior_loss /= self.dataset_size

        return prior_loss

    def disc_noise(self, disc_params): 
        with tf.variable_scope("discriminator") as scope:
            noise_loss = 0.0
            for var in disc_params.values():
                noise_ = tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())
        noise_loss /= self.dataset_size
        return noise_loss




