import tensorflow as tf


class GumbelSoftmax(tf.keras.layers.Layer):

    def __init__(self, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-10):
        """
            Sample from Gumbel(0, 1)
        """
        # noise = torch.rand(shape)
        noise = tf.random.uniform(shape)
        # noise.add_(eps).log_().neg_()
        noise = tf.negative(tf.math.log(tf.add(noise, eps)))
        # noise.add_(eps).log_().neg_()
        noise = tf.negative(tf.math.log(tf.add(noise, eps)))
        return noise

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        # uniform_samples_tensor = template_tensor.clone().uniform_()
        uniform_samples_tensor = tf.random.uniform(tf.shape(template_tensor))
        # gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        gumble_samples_tensor = - tf.math.log(eps - tf.math.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """
        Draw a sample from the Gumbel-Softmax distribution
        from torch code:
        dim = logits.size(2)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        """
        dim = logits.shape[2]
        gumble_samples_tensor = self.sample_gumbel_like(logits)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = tf.math.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    # @tf.function
    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probslibaba
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        # y shape ==>> (batch, channel, 2)
        y = self.gumbel_softmax_sample(logits, temperature)
        # y shape ==>> (batch, channel)
        y = y[:, :, 0]
        if hard:
            # from torch code
            # _, max_value_indexes = y.data.max(2, keepdim=True)
            # y_hard = logits.data.clone().zero_().scatter_(2, max_value_indexes, 1)
            # y = torch.Variable(y_hard - y.data) + y

            # tf2 code
            # in torch code, output ==>> torch.Size([batch, 192, 2, 1]), but will be used as hatten_d[:, :, 1]
            # so, we can change output as tf.Tensor([batch, 192]) or tf.Tensor([batch, 1, 1, 192])
            # Waring, in tf2 code input shape will change to tf.Tensor([batch, 192, 2])
            # todo: there may still some problems with loss computing, while use tf.identity.
            # todo: use "y = tf.stop_gradient(y_hard - y) + y" instead.
            # tried tf.Variable, but can cause exception while build graph.
            # This is an implementation of Gumbel-SoftMax.
            # Any help will be appreciated!!!
            no_grade_logits = tf.stop_gradient(logits)
            argmax = tf.argmax(no_grade_logits, axis=2, output_type=tf.int32)
            y_hard = tf.cast(argmax, dtype=tf.float32)
            # output shape ==>> (batch, channel)
            # y = tf.identity(y_hard - tf.identity(y)) + y
            no_grade = tf.stop_gradient(y_hard - y)
            y = no_grade + y
        return y

    def build(self, input_shape):
        self.built = True

    def call(self, logits, temp=1, force_hard=False, training=None, **kwargs):
        # samplesize = logits.size()
        if training and not force_hard:
            return self.gumbel_softmax(logits, temperature=temp, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=temp, hard=True)
