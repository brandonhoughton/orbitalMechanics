import tensorflow as tf
import numpy as np
from src.dataLoader.planets import get_russ_data

traj, f, planet = get_russ_data(planet = 4)

######################
# Network Parameters #
######################

inputDim = 4 #dd
hiddenDim = 50 # nh
outDim = 1 #od
lr = tf.constant(0.001)

######################

# 4 by N 
xx = np.array(traj) #xx
xxd = np.array(f)   #xxd
numExamples = xx.shape[0]

#Scale weights - using max (vector scale?)
cc  = np.max(np.abs(xx), axis = 0)
cc2 = np.max(np.abs(xxd), axis = 0)

w1 = 0.01 * np.random.randn(inputDim, hiddenDim)
w1 = w1 * (1/cc*np.ones((hiddenDim,4))).T

w2 = 0.1 * np.random.randn(hiddenDim, outDim)

print(w1)
print(w2)


# Set up computation graph
W1 = tf.Variable(w1, dtype=tf.float32)
B1 = tf.Variable(np.zeros(hiddenDim, np.float32))
W2 = tf.Variable(w2, dtype=tf.float32)
B2 = tf.Variable(np.zeros(outDim, np.float32))

#X = tf.placeholder(tf.float32, shape=(None,4))
#F = tf.placeholder(tf.float32, shape=(None,4))

# Wrap data in identity to tell tensorflow that these should not be treated as constant
X = tf.identity(tf.constant(traj, dtype='float'))
F = tf.identity(tf.constant(f, dtype='float'))

X -= tf.expand_dims(tf.reduce_mean(X, axis=1), dim=-1)

X /= tf.expand_dims(tf.reduce_max(X, axis=1), dim=-1) 
F /= tf.expand_dims(tf.reduce_max(X, axis=1), dim=-1)

# X = tf.layers.batch_normalization(X, training=True)
# F = tf.layers.batch_normalization(F, training=True)


alpha = 0.01 #Scaling factor for unit field gradient
beta  = 0.1  #Scaling factor for auto-encoder loss
# #Phi
logits = tf.sigmoid(tf.matmul(X, W1) + B1)
logits2 = tf.sigmoid(tf.matmul(logits, W2) + B2)


# head = tf.layers.dense(X, 50, activation=tf.nn.sigmoid)
# head = tf.layers.dense(head, 50, activation=tf.nn.sigmoid)
# logits2 = tf.layers.dense(head, 50)

deltaPhi = tf.gradients(logits2,[X])[0]

dotProd = tf.reduce_sum(tf.multiply(F, deltaPhi)/tf.expand_dims(tf.norm(F, axis=1)*tf.norm(deltaPhi, axis=1), dim=1), axis=1)
gradTerm = tf.square(tf.norm(deltaPhi, axis=1) - 1)
gradMag = tf.norm(deltaPhi, axis=1)
#[dotProd, gradTerm] = ([dotProd, gradTerm])

meanDotProd = tf.reduce_mean(tf.abs(dotProd))
cost = tf.reduce_mean(tf.abs(dotProd) + alpha * gradTerm)


# Setup gradients
# cost = tf.Print(cost, [X, gradTerm])
opt = tf.train.AdamOptimizer().minimize(cost)

# Setup summarys 
tf.summary.scalar('cost', cost)
tf.summary.scalar('cosineDistance', tf.reduce_mean(dotProd))
tf.summary.scalar('gradMag', tf.reduce_mean(gradMag))
tf.summary.scalar('dotProd', meanDotProd)

tf.summary.histogram('phi', logits2)
tf.summary.histogram('dotProducts', dotProd)
tf.summary.histogram('gradeints', gradTerm)

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./train/' + planet)


#####################
train_epoch = 100000
display_step = 1000000
summary_step = 100
#####################
c = 0
# avg_cost = 0
# Training step
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print("Grads for first step: ")
    # gradTermv, dotProdv = sess.run([ deltaPhi, dotProd])
    # print(gradTermv.shape, gradTermv)
    # print(dotProdv.shape, dotProdv)

    #print(sess.run([grad_W1, grad_B1, grad_W2, grad_B2, prnt]))
    # print("... Grads for first step")

    # Training cycle
    for epoch in range(train_epoch):
        # Fit training using batch data
      
        # Compute average loss
        # avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("\nEpoch", epoch)
            logits2d, l2, gradTermv, dotProdv, mdp = sess.run([logits2, deltaPhi, gradMag, dotProd, meanDotProd])
            # print("ClipVal", l1)
            print("Phi", logits2d)
            print("GradPhi", l2)
            print("GradMag", gradTermv)
            print("DotProduct", dotProdv)
            print("DotProdMag", mdp)
            print("Current error, ", c)

            # Energy function calculation
            w1, w2 = sess.run([W1, W2])
            w1probs = 1 / (1 + np.exp(np.matmul(-w1.T, xx.T))) 
            a = np.matmul(w2,np.ones((outDim,numExamples)))
            b = (a * (w1probs * (1-w1probs))).T 
            temp = np.matmul(b, w1.T) * xxd
            temp1 = sum(temp, 2)
            rowSum = np.sum(temp,axis=1) #
            f1 = sum(temp1 * temp1)

            E = np.matmul(w2.T, w1probs)  


            print("F1:\n", f1)
            print("E:\n",E)

        if (epoch+1) % summary_step == 0:
            summary = sess.run([merged])[0]
            train_writer.add_summary(summary, epoch)

        c, _ = sess.run([cost, opt])
        
        # time.sleep(0.5)



