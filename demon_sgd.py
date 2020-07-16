from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.training import training_ops
import tensorflow as tf

# SGD + momentum added decay as in
# https://arxiv.org/pdf/1910.04952.pdf
# the extra 'T' parameter is the total number of iterations
# i.e. at this point the momentum will decay to zero
class DemonSGD(SGD):
    def __init__(
            self, T,
            learning_rate=0.01,
            momentum=0.0,
            nesterov=False,
            name="SGD",
            **kwargs):
        super(DemonSGD, self).__init__(learning_rate, momentum, nesterov, name, **kwargs)
        self._set_hyper("T", T)

    # see Algorithm 1 in the paper
    def _momentum_decay(self, beta_init):
        if self._iterations is not None:
            decay_rate = tf.cast(1.0 - self._iterations / self._get_hyper("T", tf.int64), tf.float32)
        else:
            decay_rate = 1.0
        beta_decay = beta_init * decay_rate
        beta = beta_decay / ((1.0 - beta_init) + beta_decay)
        return beta

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        if self._momentum:
            momentum_var = self.get_slot(var, "momentum")
            momentum_coeff = self._momentum_decay(coefficients["momentum"])
            return training_ops.resource_apply_keras_momentum(
                var.handle,
                momentum_var.handle,
                coefficients["lr_t"],
                grad,
                momentum_coeff,
                use_locking=self._use_locking,
                use_nesterov=self.nesterov)
        else:
            return training_ops.resource_apply_gradient_descent(
                var.handle, coefficients["lr_t"], grad, use_locking=self._use_locking)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        # This method is only needed for momentum optimization.
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = ((apply_state or {}).get((var_device, var_dtype))
                        or self._fallback_apply_state(var_device, var_dtype))

        momentum_var = self.get_slot(var, "momentum")
        momentum_coeff = self._momentum_decay(coefficients["momentum"])
        return training_ops.resource_sparse_apply_keras_momentum(
            var.handle,
            momentum_var.handle,
            coefficients["lr_t"],
            grad,
            indices,
            momentum_coeff,
            use_locking=self._use_locking,
            use_nesterov=self.nesterov)

    def get_config(self):
        config = super(DemonSGD, self).get_config()
        config.update({
            "T": self._serialize_hyperparameter("T")
        })
        return config
