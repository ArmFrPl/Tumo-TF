
with tf.Graph().as_default() as g:
    with tf.Session() as sess:
        input_data = [[10, 7], [5, 4], [8, 3], [2, 8], [4, 8], [9, 1], [6, 8], [1, 7], [5, 3], [7, 6]]
        output_data = [[97], [71], [74], [78], [90], [68], [95], [65], [70], [82]]
        input_plch = tf.placeholder(tf.float32)
        output_plch = tf.placeholder(tf.float32)
        feed_dict = {input_plch:input_data, output_plch:output_data}
        max_input = tf.reduce_max(input_plch, 0)
        normalized_input = tf.divide(input_plch, max_input)
        
        max_output = tf.reduce_max(output_plch, 0)
        normalized_output = output_plch/max_output
        #print(sess.run(normalized_input, feed_dict = feed_dict))
        #print(sess.run(normalized_output, feed_dict = feed_dict))
        #print(sess.run(output_plch, feed_dict = feed_dict))
        #print(sess.run(normalized_output, feed_dict = feed_dict))
        
        num_neurons_1 = 2
        num_neurons_2 = 3
        num_neurons_3 = 1
        
        weights_1 = tf.Variable(tf.zeros([num_neurons_1, num_neurons_2]))
        bias_1 = tf.Variable(tf.zeros(num_neurons_2))
        weighted_sums_1 = tf.matmul(normalized_input, weights_1) + bias_1
        activation_1 = tf.sigmoid(weighted_sums_1)
        
        weights_2 = tf.Variable(tf.zeros([num_neurons_2, num_neurons_3]))
        bias_2= tf.Variable(tf.zeros(num_neurons_3))
        weighted_sums_2 = tf.matmul(activation_1, weights_2) + bias_2
        activation_2 = tf.sigmoid(weighted_sums_2)
        
        loss = tf.reduce_sum((activation_2 - output_plch)**2)/10
        optimizer = tf.train.GradientDescentOptimizer(0.05)
        train_step = optimizer.minimize(loss)
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            sess.run(train_step, feed_dict = feed_dict)
            #print(sess.run(loss, feed_dict = feed_dict))
            #print(sess.run(weights_1, feed_dict = feed_dict))
            #print(sess.run(bias_1,feed_dict = feed_dict))
    
        #print(sess.run(loss, feed_dict = feed_dict))
        #print(sess.run(activation_2, feed_dict = feed_dict))
        #print(sess.run(max_output, feed_dict = feed_dict))