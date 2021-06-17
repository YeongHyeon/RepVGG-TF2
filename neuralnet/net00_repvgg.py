import os
import numpy as np
import tensorflow as tf
import source.utils as utils
import whiteboxlayer.layers as wbl
import whiteboxlayer.extensions.utility as wblu

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")

        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.ksize = kwargs['ksize']
        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.variables = {}

        self.__model = Neuralnet(\
            who_am_i="RepVGG", **kwargs, \
            filters=[1, 32, 64, 128])

        dummy = tf.zeros((1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32)
        self.__model.forward(x=dummy, training=True, verbose=True)
        self.__model.forward(x=dummy, training=False, verbose=True)

        self.__init_propagation(path=self.path_ckpt)

    def __init_propagation(self, path):

        self.summary_writer = tf.summary.create_file_writer(self.path_ckpt)

        self.variables['trainable'] = []
        ftxt = open("list_parameters.txt", "w")
        for key in list(self.__model.layer.parameters.keys()):
            trainable = self.__model.layer.parameters[key].trainable
            text = "T: " + str(key) + str(self.__model.layer.parameters[key].shape)
            if(trainable):
                self.variables['trainable'].append(self.__model.layer.parameters[key])
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.save_params()

    def __loss(self, y, y_hat):

        entropy_b = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat)
        entropy = tf.math.reduce_mean(entropy_b)

        return {'entropy_b': entropy_b, 'entropy': entropy}

    @tf.autograph.experimental.do_not_convert
    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['y']

        with tf.GradientTape() as tape:
            logit, y_hat = self.__model.forward(x=x, training=training, verbose=False)
            losses = self.__loss(y=y, y_hat=logit)

        if(training):
            gradients = tape.gradient(losses['entropy'], self.variables['trainable'])
            self.optimizer.apply_gradients(zip(gradients, self.variables['trainable']))

            with self.summary_writer.as_default():
                tf.summary.scalar('%s/entropy' %(self.__model.who_am_i), losses['entropy'], step=iteration)

        return {'y_hat':y_hat, 'losses':losses}

    def save_params(self, model='base', tflite=False):

        if(tflite):
            # https://github.com/tensorflow/tensorflow/issues/42818
            conc_func = self.__model.__call__.get_concrete_function(\
                tf.TensorSpec(shape=(1, self.dim_h, self.dim_w, self.dim_c), dtype=tf.float32))
            converter = tf.lite.TFLiteConverter.from_concrete_functions([conc_func])

            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]

            tflite_model = converter.convert()

            with open('model.tflite', 'wb') as f:
                f.write(tflite_model)
        else:
            vars_to_save = self.__model.layer.parameters.copy()
            vars_to_save["optimizer"] = self.optimizer

            ckpt = tf.train.Checkpoint(**vars_to_save)
            ckptman = tf.train.CheckpointManager(ckpt, directory=os.path.join(self.path_ckpt, model), max_to_keep=1)
            ckptman.save()

    def load_params(self, model):

        vars_to_load = self.__model.layer.parameters.copy()
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(os.path.join(self.path_ckpt, model))
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

class Neuralnet(tf.Module):

    def __init__(self, **kwargs):
        super(Neuralnet, self).__init__()

        self.who_am_i = kwargs['who_am_i']
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.ksize = kwargs['ksize']
        self.num_class = kwargs['num_class']
        self.filters = kwargs['filters']

        self.layer = wbl.Layers()

        self.forward = tf.function(self.__call__)

    @tf.function
    def __call__(self, x, training=False, verbose=False):

        logit = self.__nn(x=x, training=training, name=self.who_am_i, verbose=verbose)
        y_hat = tf.nn.softmax(logit, name="y_hat")

        return logit, y_hat

    def __nn(self, x, training, name='neuralnet', verbose=True):

        att = None
        for idx, _ in enumerate(self.filters[:-1]):
            if(idx == 0): continue
            x = self.__repvgg(x=x, ksize=self.ksize, filter_in=self.filters[idx-1], filter_out=self.filters[idx], stride=2, \
                activation='relu', name='%s-%d_repvgg1' %(name, idx), training=training, verbose=verbose)
            x = self.__repvgg(x=x, ksize=self.ksize, filter_in=self.filters[idx], filter_out=self.filters[idx], stride=1, \
                activation='relu', name='%s-%d_repvgg2' %(name, idx), training=training, verbose=verbose)

        x = tf.math.reduce_mean(x, axis=(1, 2))
        x = self.layer.fully_connected(x=x, c_out=self.num_class, \
                activation=None, name="%s-clf" %(name), verbose=verbose)

        return x

    def __repvgg(self, x, ksize, filter_in, filter_out, stride=1, \
        activation='relu', name="", training=False, verbose=False):

        if(training):
            x_main = self.layer.conv2d(x=x, stride=stride, \
                filter_size=[ksize, ksize, filter_in, filter_out], \
                batch_norm=True, activation='relu', name='%s_main' %(name), verbose=verbose)
            x_sub = self.layer.conv2d(x=x, stride=stride, \
                filter_size=[1, 1, filter_in, filter_out], \
                batch_norm=True, activation='relu', name='%s_sub' %(name), verbose=verbose)

            if(stride == 1):
                x_id = self.layer.batch_normalization(x=x, trainable=True, name='%s_id_bn' %(name), verbose=verbose)
                x = x_main + x_sub + x_id
            else:
                x = x_main + x_sub
        else:
            x = self.__fuse_to_deploy(x=x, stride=stride, name=name)

        return self.layer.activation(x, activation=activation, name='%s_act' %(name))

    def __fuse_to_deploy(self, x, stride=1, \
        dilations=[1, 1, 1, 1], padding='SAME', name=""):

        main_w = self.layer.parameters['%s_main_w' %(name)]
        main_mean, main_variance, main_ofs, main_sce = self.__bn_extraction(x=x, name="%s_main_bn" %(name))
        main_w, main_b = self.__fuse_w_bn(main_w, main_mean, main_variance, main_ofs, main_sce)

        sub_w = self.layer.parameters['%s_sub_w' %(name)]
        sub_mean, sub_variance, sub_ofs, sub_sce = self.__bn_extraction(x=x, name="%s_sub_bn" %(name))
        sub_wp = tf.pad(sub_w, paddings=[[1, 1], [1, 1], [0, 0], [0, 0]], constant_values=0)
        sub_w, sub_b = self.__fuse_w_bn(sub_wp, sub_mean, sub_variance, sub_ofs, sub_sce)

        w = main_w + sub_w
        b = main_b + sub_b

        if(stride == 1):
            list_shape = main_w.get_shape().as_list()
            id_w = np.zeros(list_shape)
            idx_k, num_c = int(list_shape[0]/2), list_shape[-2]
            group = num_c // 2
            for idx_c in range(num_c):
                id_w[idx_k, idx_k, idx_c, idx_c % group] = 1
            id_w = tf.convert_to_tensor(id_w, dtype=tf.float32)
            id_mean, id_variance, id_ofs, id_sce = self.__bn_extraction(x=x, name="%s_id_bn" %(name))
            id_w, id_b = self.__fuse_w_bn(id_w, id_mean, id_variance, id_ofs, id_sce)

            w = w + id_w
            b = b + id_b

        wx = tf.nn.conv2d(
            input=x,
            filters=w,
            strides=[1, stride, stride, 1],
            padding=padding,
            data_format='NHWC',
            dilations=dilations,
            name='%s_deploy_conv' %(name)
        )

        return tf.math.add(wx, b, name='%s_deploy_add' %(name))

    def __bn_extraction(self, x, name=""):

        mean, variance = tf.nn.moments(x=x, axes=[0], keepdims=True, name="%s_mmt" %(name))
        ofs, sce = self.layer.parameters['%s_ofs' %(name)], self.layer.parameters['%s_sce' %(name)]

        return mean, variance, ofs, sce

    def __fuse_w_bn(self, w, mean, variance, offset, scale):

        mean = tf.math.reduce_mean(mean, axis=(1, 2, 3))
        variance = tf.math.reduce_mean(variance, axis=(1, 2, 3))
        std = (variance + 1e-12)**(0.5)
        t = scale / std
        return w * t, offset - mean * scale / std
