import tensorflow as tf
import math

length = 20
channels = 8
min_timescale = 1.0
max_timescale = 1.0e4
start_index = 0

position = tf.compat.v1.to_float(tf.range(length) + start_index)
num_timescales = channels // 2
log_timescale_increment = (
  math.log(float(max_timescale) / float(min_timescale)) /
  (tf.compat.v1.to_float(num_timescales) - 1))
inv_timescales = min_timescale * tf.exp(
  tf.compat.v1.to_float(tf.range(num_timescales)) * -log_timescale_increment)
scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
signal = tf.concat([tf.math.sin(scaled_time), tf.cos(scaled_time)], axis=1)
signal = tf.pad(signal, [[0, 0], [0, tf.math.floormod(channels, 2)]])
signal = tf.reshape(signal, [1, length, channels])

print(signal)