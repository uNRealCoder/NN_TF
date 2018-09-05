import pickle
import os
import numpy as np
import tensorflow as tf  
import urllib.request as request  

def create_train_model(hidden_nodes, num_iters,num_samples):

    # Reset the graph
    tf.reset_default_graph()

    # Placeholders for input and output data
    X = tf.placeholder(shape=(num_samples, 16), dtype=tf.float64, name='X')
    y = tf.placeholder(shape=(num_samples, 2), dtype=tf.float64, name='y')

    # Variables for two group of weights between the three layers of the network
    W1 = tf.Variable(np.random.rand(16, hidden_nodes), dtype=tf.float64)
    W2 = tf.Variable(np.random.rand(hidden_nodes, 2), dtype=tf.float64)

    # Create the neural net graph
    A1 = tf.sigmoid(tf.matmul(X, W1))
    y_est = tf.sigmoid(tf.matmul(A1, W2))

    # Define a loss function
    deltas = tf.square(y_est - y)
    loss = tf.reduce_sum(deltas)

    # Define a train operation to minimize the loss
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # Initialize variables and run session
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Go through num_iters iterations
    for i in range(num_iters):
        sess.run(train, feed_dict={X: X_train, y: Y_train})
        loss_plot[hidden_nodes].append(sess.run(loss, feed_dict={X: X_train, y: Y_train}))
        weights1 = sess.run(W1)
        weights2 = sess.run(W2)

    print("loss (hidden nodes: %d, iterations: %d): %.2f" % (hidden_nodes, num_iters, loss_plot[hidden_nodes][-1]))
    sess.close()
    return weights1, weights2



if __name__=="__main__":
    print("Initialization");

    X_Data=pickle.load(open('X','rb'));
    Y_Data=pickle.load(open('Y','rb'));
    n=len(X_Data);
    split=int(0.8*n);
    X_train=X_Data[0:split];
    Y_train=Y_Data[0:split];
    X_test=X_Data[split:n]
    Y_test=Y_Data[split:n];

    print("Training Starts");

    num_hidden_nodes = [5, 10, 20, 40]  
    loss_plot = {5: [], 10: [], 20: [], 40:[]}  
    weights1 = {5: None, 10: None, 20: None, 40:None}  
    weights2 = {5: None, 10: None, 20: None, 40:None}
    num_iters = 2
    for hidden_nodes in num_hidden_nodes:  
        weights1[hidden_nodes], weights2[hidden_nodes] = create_train_model(hidden_nodes, num_iters,split)
        print(hidden_nodes)

    print("Testing Starts")
    X = tf.placeholder(shape=(n-split, 16), dtype=tf.float64, name='X')  
    Y = tf.placeholder(shape=(n-split, 2), dtype=tf.float64, name='Y')

    for hidden_nodes in num_hidden_nodes:

        # Forward propagation
        W1 = tf.Variable(weights1[hidden_nodes])
        W2 = tf.Variable(weights2[hidden_nodes])
        A1 = tf.sigmoid(tf.matmul(X, W1))
        y_est = tf.sigmoid(tf.matmul(A1, W2))
        # Calculate the predicted outputs
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            y_est_np = sess.run(y_est, feed_dict={X: X_test, Y: Y_test})
            numbs = 0;
            for estimate,correct in zip(y_est_np,Y_test):
                if(round(estimate[0])==correct[0] and round(estimate[1])==correct[1]):
                   numbs = numbs+1;
            accuracy = 100 * (numbs) / (n-split)
            print('Network architecture 16-%d-2 accuracy: %.2f%%' % (hidden_nodes, accuracy))
