from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf

# Adam + momentum decay added as in
# https://arxiv.org/pdf/1910.04952.pdf
# the extra 'T' parameter is the total number of iterations
# i.e. at this point the momentum will decay to zero
class DemonAdam(Adam):
    def __init__(
            self, T,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            amsgrad=False,
            name='DemonAdam',
            **kwargs):
        super(DemonAdam, self).__init__(learning_rate, beta_1, beta_2, epsilon, amsgrad, name, **kwargs)
        self._set_hyper("T", T)

    # see Algorithm 2 in the paper
    def _momentum_decay(self, beta_init):
        if self.iterations is not None:
            decay_rate = tf.cast(1.0 - self.iterations / self._get_hyper("T", tf.int64), tf.float32)
        else:
            decay_rate = 1.0
        beta_decay = beta_init * decay_rate
        beta = beta_decay / ((1.0 - beta_init) + beta_decay)
        return beta

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super(DemonAdam, self)._prepare_local(var_device, var_dtype, apply_state)

        local_step = math_ops.cast(self.iterations + 1, var_dtype)
        beta_init = array_ops.identity(self._get_hyper('beta_1', var_dtype))
        beta_1_t = self._momentum_decay(beta_init)
        beta_2_t = array_ops.identity(self._get_hyper('beta_2', var_dtype))
        beta_1_power = math_ops.pow(beta_1_t, local_step)
        beta_2_power = math_ops.pow(beta_2_t, local_step)
        lr = (apply_state[(var_device, var_dtype)]['lr_t'] *
              (math_ops.sqrt(1 - beta_2_power) / (1 - beta_1_power)))
        apply_state[(var_device, var_dtype)].update(
            dict(
                lr=lr,
                epsilon=ops.convert_to_tensor_v2(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t))

    def get_config(self):
        config = super(DemonAdam, self).get_config()
        config.update({
            "T": self._serialize_hyperparameter("T")
        })
        return config
