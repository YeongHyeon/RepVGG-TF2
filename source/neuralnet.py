import os
import tensorflow as tf
import source.layers as lay

class CNN(object):

    def __init__(self, height, width, channel, num_class, \
        ksize, radix=4, kpaths=4, learning_rate=1e-3, ckpt_dir='./Checkpoint'):

        print("\nInitializing Short-RegNet...")
        self.height, self.width, self.channel, self.num_class = height, width, channel, num_class
        self.ksize, self.learning_rate = ksize, learning_rate
        self.radix, self.kpaths = radix, kpaths
        self.ckpt_dir = ckpt_dir

        self.customlayers = lay.Layers()
        self.model(tf.zeros([1, self.height, self.width, self.channel]), verbose=True)

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        self.summary_writer = tf.summary.create_file_writer(self.ckpt_dir)

    def step(self, x, y, iteration=0, train=False):

        with tf.GradientTape() as tape:
            logits = self.model(x, train=train, verbose=False)
            smce = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
            loss = tf.math.reduce_mean(smce)

        score = self.customlayers.softmax(logits)
        pred = tf.argmax(score, 1)
        correct_pred = tf.equal(pred, tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        if(train):
            gradients = tape.gradient(loss, self.customlayers.params_trainable)
            self.optimizer.apply_gradients(zip(gradients, self.customlayers.params_trainable))

            with self.summary_writer.as_default():
                tf.summary.scalar('RegVGG/loss', loss, step=iteration)
                tf.summary.scalar('RegVGG/accuracy', accuracy, step=iteration)

        return loss, accuracy, score

    def save_params(self):

        vars_to_save = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_save[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_save["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_save)
        ckptman = tf.train.CheckpointManager(ckpt, directory=self.ckpt_dir, max_to_keep=3)
        ckptman.save()

    def load_params(self):

        vars_to_load = {}
        for idx, name in enumerate(self.customlayers.name_bank):
            vars_to_load[self.customlayers.name_bank[idx]] = self.customlayers.params_trainable[idx]
        vars_to_load["optimizer"] = self.optimizer

        ckpt = tf.train.Checkpoint(**vars_to_load)
        latest_ckpt = tf.train.latest_checkpoint(self.ckpt_dir)
        status = ckpt.restore(latest_ckpt)
        status.expect_partial()

    def model(self, x, train=False, verbose=False):

        if(verbose): print("input", x.shape)

        conv1_1 = self.repvgg(x, \
            ksize=self.ksize, inchannel=self.channel, outchannel=16, stride_size=2, \
            name="conv1_1", train=train, verbose=verbose)
        conv1_2 = self.repvgg(conv1_1, \
            ksize=self.ksize, inchannel=16, outchannel=16, stride_size=2, \
            name="conv1_2", train=train, verbose=verbose)

        conv2_1 = self.repvgg(conv1_2, \
            ksize=self.ksize, inchannel=16, outchannel=32, stride_size=2, \
            name="conv2_1", train=train, verbose=verbose)
        conv2_2 = self.repvgg(conv2_1, \
            ksize=self.ksize, inchannel=32, outchannel=32, stride_size=1, \
            name="conv2_2", train=train, verbose=verbose)

        conv3_1 = self.repvgg(conv2_2, \
            ksize=self.ksize, inchannel=32, outchannel=64, stride_size=2, \
            name="conv3_1", train=train, verbose=verbose)
        conv3_2 = self.repvgg(conv3_1, \
            ksize=self.ksize, inchannel=64, outchannel=64, stride_size=1, \
            name="conv3_2", train=train, verbose=verbose)

        [n, h, w, c] = conv3_2.shape
        flat = tf.compat.v1.reshape(conv3_2, shape=[-1, h*w*c], name="flat")
        if(verbose):
            num_param_fe = self.customlayers.num_params
            print("flat", flat.shape)

        fc1 = self.customlayers.fullcon(flat, \
            self.customlayers.get_weight(vshape=[h*w*c, self.num_class], name="fullcon1"))
        if(verbose):
            print("fullcon1", fc1.shape)
            print("\nNum Parameter")
            print("Feature Extractor : %d" %(num_param_fe))
            print("Classifier        : %d" %(self.customlayers.num_params - num_param_fe))
            print("Total             : %d" %(self.customlayers.num_params))

        return fc1

    def repvgg(self, input, ksize, inchannel, outchannel, stride_size=1, \
        name="", train=False, verbose=False):

        branch_main = self.customlayers.conv2d(input, \
            self.customlayers.get_weight(vshape=[ksize, ksize, inchannel, outchannel], name="%s_main" %(name)), \
            stride_size=stride_size, padding='SAME')
        branch_main_bn = self.customlayers.batch_norm(branch_main, name="%s_main_bn" %(name))
        branch_main_act = self.customlayers.elu(branch_main_bn)

        branch_sub = self.customlayers.conv2d(input, \
            self.customlayers.get_weight(vshape=[ksize, ksize, inchannel, outchannel], name="%s_sub" %(name)), \
            stride_size=stride_size, padding='SAME')
        branch_sub_bn = self.customlayers.batch_norm(branch_sub, name="%s_sub_bn" %(name))
        branch_sub_act = self.customlayers.elu(branch_sub_bn)

        if(train):
            if(stride_size == 1):
                output = branch_main_act + branch_sub_act + input
            else:
                output = branch_main_act + branch_sub_act
        else:
            output = branch_main_act

        if(verbose): print(name, output.shape)
        return output
