from jax import random
from neural_tangents import stax

init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(512), stax.Relu(),
    stax.Dense(512), stax.Relu(),
    stax.Dense(1)
)

key1, key2 = random.split(random.PRNGKey(1))
x1 = random.normal(key1, (10, 100))
x2 = random.normal(key2, (20, 100))

kernel = kernel_fn(x1, x2, 'nngp')


# Get kernel of a single type
nngp = kernel_fn(x1, x2, 'nngp') # (10, 20) jnp.ndarray
ntk = kernel_fn(x1, x2, 'ntk') # (10, 20) jnp.ndarray

# Get kernels as a namedtuple
both = kernel_fn(x1, x2, ('nngp', 'ntk'))
both.nngp == nngp  # True
both.ntk == ntk  # True

# Unpack the kernels namedtuple
nngp, ntk = kernel_fn(x1, x2, ('nngp', 'ntk'))


kernel = kernel_fn(x1, x2)
kernel = kernel_fn(kernel)
print(kernel.nngp)

import neural_tangents as nt

x_train, x_test = x1, x2
y_train = random.uniform(key1, shape=(10, 1))  # training targets

predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train,
                                                      y_train)

y_test_nngp = predict_fn(x_test=x_test, get='nngp')
# (20, 1) jnp.ndarray test predictions of an infinite Bayesian network

y_test_ntk = predict_fn(x_test=x_test, get='ntk')
# (20, 1) jnp.ndarray test predictions of an infinite continuous
# gradient descent trained network at convergence (t = inf)

# Get predictions as a namedtuple
both = predict_fn(x_test=x_test, get=('nngp', 'ntk'))
both.nngp == y_test_nngp  # True
both.ntk == y_test_ntk  # True

# Unpack the predictions namedtuple
y_test_nngp, y_test_ntk = predict_fn(x_test=x_test, get=('nngp', 'ntk'))


from neural_tangents import stax

def WideResnetBlock(channels, strides=(1, 1), channel_mismatch=False):
  Main = stax.serial(
      stax.Relu(), stax.Conv(channels, (3, 3), strides, padding='SAME'),
      stax.Relu(), stax.Conv(channels, (3, 3), padding='SAME'))
  Shortcut = stax.Identity() if not channel_mismatch else stax.Conv(
      channels, (3, 3), strides, padding='SAME')
  return stax.serial(stax.FanOut(2),
                     stax.parallel(Main, Shortcut),
                     stax.FanInSum())

def WideResnetGroup(n, channels, strides=(1, 1)):
  blocks = []
  blocks += [WideResnetBlock(channels, strides, channel_mismatch=True)]
  for _ in range(n - 1):
    blocks += [WideResnetBlock(channels, (1, 1))]
  return stax.serial(*blocks)

def WideResnet(block_size, k, num_classes):
  return stax.serial(
      stax.Conv(16, (3, 3), padding='SAME'),
      WideResnetGroup(block_size, int(16 * k)),
      WideResnetGroup(block_size, int(32 * k), (2, 2)),
      WideResnetGroup(block_size, int(64 * k), (2, 2)),
      stax.AvgPool((8, 8)),
      stax.Flatten(),
      stax.Dense(num_classes, 1., 0.))

init_fn, apply_fn, kernel_fn = WideResnet(block_size=4, k=1, num_classes=10)