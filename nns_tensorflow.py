from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import time

#assume the dataset is imported and X and Y are created
#Scale all values. X is the input and Y is the output
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

#Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#Reshape the data. This is an example with 5 inputs and 1 output
#X_train = X_train.reshape([-1,1,5])
#Y_train = Y_train.reshape([-1,1])
#X_test = X_test.reshape([-1,1,5])
#Y_test = Y_test.reshape([-1,1])

#set hyper parameters
hidden_nodes = 3
output_nodes = 1
batch_size = 100
samples = Y_train.size
#set the number of epoch
epochs = 10

###############################################
#create a feed forward back propagation NN - BPNN with tensorflow
import tensorflow as tf

x = tf.placeholder('float32', [None, 5])
y = tf.placeholder('float32', [None, 1])

#define neural network
def BPNN(x):
    hidden_layer = {
        'weights': tf.Variable(tf.random_normal([5, hidden_nodes])),
        'biases': tf.Variable(tf.random_normal([hidden_nodes]))
    }
    output_layer = {
        'weights': tf.Variable(tf.random_normal([hidden_nodes, output_nodes])),
        'biases': tf.Variable(tf.random_normal([output_nodes]))
    }

    
    # (input_data * weights) + biases
    l1 = tf.add(tf.matmul(x, hidden_layer['weights']), hidden_layer['biases'])

    l1 = tf.nn.tanh(l1)

    output = tf.matmul(l1, output_layer['weights']) + output_layer['biases']

    return output

start = time.time()

#set the learning parameters
def nn_train(x):
    prediction = BPNN(x)
    mae = tf.reduce_sum(tf.abs(prediction - y))/(samples)
    rmse = tf.sqrt(tf.reduce_sum(tf.pow(prediction - y, 2))/(samples))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mae)

    #run the session
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(samples):
                batch_x = X_train[_*batch_size:(_+1)*batch_size]
                batch_y = Y_train[_*batch_size:(_+1)*batch_size]
                _, c = sess.run([train_step, mae], feed_dict={x: batch_x, y: batch_y})
          
                epoch_loss += c
            print('At epoch: ', epoch, ' mae: ', epoch_loss)
        Y_pred = sess.run(prediction, feed_dict = {x: X_test, y: Y_test})
        
        #compute and print the loss
        mae_test = mae.eval({x: X_test, y: Y_test})
        rmse_test = rmse.eval({x: X_test, y: Y_test})

        print("Test MAE = ", mae_test)
        print("Test RMSE = ", rmse_test)
        print('compilation duration : ', time.time() - start) 
        
        #invert scaling
        pred_initial = scaler.inverse_transform(Y_pred)
        test_initial = scaler.inverse_transform(Y_test)

nn_train(x)



###############################################
#create a recurrent NN - LSTM with tensorflow
from tensorflow.python.ops import rnn, rnn_cell

x = tf.placeholder('float32', [None, 1, 5])
y = tf.placeholder('float32', [None, 1])

#define neural network
def RNN(x):
    output_layer = {
        'weights': tf.Variable(tf.random_normal([hidden_nodes, output_nodes])),
        'biases': tf.Variable(tf.random_normal([output_nodes]))
    }
    
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, 5])
    x = tf.split(0, 1, x)
    
    lstm_cell = rnn_cell.BasicLSTMCell(hidden_nodes)
    outputs, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)
    output = tf.matmul(outputs[-1], output_layer['weights']) + output_layer['biases']
    return output

start = time.time()

#set the learning parameters
def rnn_train(x):
    prediction = RNN(x)
    mae = tf.reduce_sum(tf.abs(prediction - y))/(samples)
    rmse = tf.sqrt(tf.reduce_sum(tf.pow(prediction - y, 2))/(samples))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(mae)

    # cycles feed forward + backprop
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_loss = 0
            for _ in range(samples):
                batch_x = X_train[_*batch_size:(_+1)*batch_size]
                batch_y = Y_train[_*batch_size:(_+1)*batch_size]
                #batch_x = batch_x.reshape((batch_size, n_chunks, chunk_size))  
                _, c = sess.run([train_step, mae], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

            print('At epoch: ', epoch, ' mae: ', epoch_loss)
        Y_pred = sess.run(prediction, feed_dict = {x: X_test, y: Y_test})

        #compute and print the loss
        mae_test = mae.eval({x: X_test, y: Y_test})
        rmse_test = rmse.eval({x: X_test, y: Y_test})

        print("test MAE = ", mae_test)
        print("Test RMSE = ", rmse_test)
        print('compilation duration : ', time.time() - start) 
        
        #invert scaling
        pred_initial = scaler.inverse_transform(Y_pred)
        test_initial = scaler.inverse_transform(Y_test)

rnn_train(x)