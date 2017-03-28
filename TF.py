import tensorflow as tf

tf.constant("Hello World!", dtype = None, name  = "Const")

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a*b

sess = tf.Session()

print(sess.run(c))
sess.close()
