import tensorflow as tf

opt = tf.keras.optimizers.Adam()
x = tf.Variable(0.)
loss_fn = lambda: (x - 5.) ** 2
results = opt.minimize(loss_fn, [x])


# var1, var2 = tf.Variable(8.), tf.Variable(4.)
# # Create an optimizer with the desired parameters.
# opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# # `loss` is a callable that takes no argument and returns the value
# # to minimize.
# loss = lambda: 3 * var1 * var1 + 2 * var2 * var2
# # In graph mode, returns op that minimizes the loss by updating the listed
# # variables.
# opt_op = opt.minimize(loss, var_list=[var1, var2])
# # opt_op.run()
# # In eager mode, simply call minimize to update the list of variables.
# # opt.minimize(loss, var_list=[var1, var2])
