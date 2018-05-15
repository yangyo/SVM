import tensorflow as tf
import numpy as np
import math
from matplotlib import pyplot as plt
from tensorflow import flags

class SVM():
    def __init__(self):
        self.x=tf.placeholder('float',shape=[None,2],name='x_batch')
        self.y=tf.placeholder('float',shape=[None,1],name='y_batch')
        self.x_min=-100
        self.x_max=100
        self.y_min=-2
        self.y_max=2
#        self.sess=tf.Session()

    def creat_dataset(self,size, n_dim=2, center=0, dis=2, scale=1, one_hot=False):
        center1 = (np.random.random(n_dim) + center - 0.5) * scale + dis
        center2 = (np.random.random(n_dim) + center - 0.5) * scale - dis
        cluster1 = (np.random.randn(size, n_dim) + center1) * scale
        cluster2 = (np.random.randn(size, n_dim) + center2) * scale
        x_data = np.vstack((cluster1, cluster2)).astype(np.float32)
        y_data = np.array([1] * size + [-1] * size)
        indices = np.random.permutation(size * 2)
        data, labels = x_data[indices], y_data[indices]
        labels=np.reshape(labels,(-1,1))
        if not one_hot:
            return data, labels
        labels = np.array([[0, 1] if label == 1 else [1, 0] for label in labels], dtype=np.int8)
        return data, labels

    @staticmethod
    def get_base(self,_nx, _ny):
        _xf = np.linspace(self.x_min, self.x_max, _nx)
        _yf = np.linspace(self.y_min, self.y_max, _ny)
        n_xf, n_yf = np.meshgrid(_xf, _yf)
        return _xf, _yf,np.c_[n_xf.ravel(), n_yf.ravel()]


    def predict(self,y_data):

        correct = tf.equal(self.y_predict_value, y_data)

        precision=tf.reduce_mean(tf.cast(correct, tf.float32))

        precision_value=self.sess.run(precision)
        return precision_value, self.y_predict_value


    def shuffle(self,epoch,batch,x_data,y_data):
        for i in range(epoch):
            shuffle_index=np.random.permutation(y_data.shape[0])
            x_data1, y_data1 = x_data[shuffle_index], y_data[shuffle_index]
            batch_per_epoch = math.ceil(y_data.shape[0]*2 / batch)
            for b in range(batch_per_epoch):
                if (b*batch+batch)>y_data.shape[0]:
                    a,b = b*batch, y_data.shape[0]
                else:
                    a,b = b*batch, b*batch+batch

                data, labels = x_data1[a:b,:], y_data1[a:b,:]
                yield data, labels


    def train(self,epoch,x_data,n_dim,y_data,x_edata,y_edata):

        w = tf.Variable(np.ones([n_dim,1]), dtype=tf.float32, name="w_v")
        b = tf.Variable(0., dtype=tf.float32, name="b_v")

        y_pred =tf.matmul(self.x,w)+b

        cost = tf.nn.l2_loss(w)+tf.reduce_sum(tf.maximum(1-self.y*y_pred,0))
        train_step = tf.train.AdamOptimizer(0.01).minimize(cost)

        y_predict =tf.sign( y_pred)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
                sess.run(init)
                shuffle= self.shuffle(epoch,100,x_data,y_data)
                for i, (x_batch, y_batch) in enumerate(shuffle):

        #            index=np.random.permutation(y_data.shape[0])
        #            x_data1, y_data1 = x_data[index], y_data[index]

                    sess.run(train_step,feed_dict={self.x:x_batch,self.y:y_batch})

                    if i%1000==0:
                        self.y_predict_value,self.w_value,self.b_value,cost_value=sess.run([y_predict,w,b,cost],feed_dict={self.x:x_data,self.y:y_data})
                        print('step= %d  ,  cost=%f '%(i, cost_value))

                        y_pre = np.sign(np.matmul(x_edata,self.w_value)+self.b_value)
                        correct = np.equal(y_pre, y_edata)

                        precision=np.mean(np.cast[ 'float32'](correct))

#                        precision_value=sess.run(precision)
                        print('eval= %d'%precision)


    def drawresult(self,x_data):

        self.x_min, self.x_max = np.min(x_data[:,0]), np.max(x_data[:,0])
        self.y_min, self.y_max = np.min(x_data[:,1]), np.max(x_data[:,1])
        x_min=self.x_min
        x_max=self.x_max
        y_min = self.x_min
        y_max = self.x_max
        x_padding = max(abs(x_min), abs(x_max)) * FLAGS.padding
        y_padding = max(abs(y_min), abs(y_max)) * FLAGS.padding
        x_min -= x_padding
        x_max += x_padding
        y_min -= y_padding
        y_max += y_padding

#        self.x_min, self.y_min = np.minimum.reduce(x_data,axis=0) -2
#        self.x_max, self.y_max = np.maximum.reduce(x_data,axis=0) +2

        xf, yf , matrix_= self.get_base(self,200, 200)

        print(self.w_value,self.b_value)
        z=np.sign(np.matmul(matrix_,self.w_value)+self.b_value).reshape((200,200))
        plt.pcolormesh(xf, yf, z, cmap=plt.cm.Paired)

        ypv = self.y_predict_value
        y_0 = np.where(ypv==1)
        y_1 = np.where(ypv==-1)
        plt.scatter(x_data[y_0,0], x_data[y_0,1],  c='g')
        plt.scatter(x_data[y_1,0], x_data[y_1,1],  c='r')

        plt.axis([x_min,x_max,y_min ,y_max])
#        plt.contour(xf, yf, z)
        plt.show()

if __name__ == '__main__':

    flags.DEFINE_integer('epoch', 1000, "number of epoch")
    flags.DEFINE_float('lr', 0.01, "learning rate")
    flags.DEFINE_integer('padding', 0.1, "padding")
    flags.DEFINE_integer('batch', 100, "batch size")
    FLAGS = flags.FLAGS

    svm=SVM()
    n_dim=2
    x_data,y_data=svm.creat_dataset(size=100, n_dim=n_dim, center=0, dis=4,  one_hot=False)
    x_edata,y_edata=svm.creat_dataset(size=100, n_dim=n_dim, center=0, dis=4,  one_hot=False)


    svm.train(FLAGS.epoch,x_data,n_dim,y_data,x_edata,y_edata)
    #precision_value,y_predict_value=svm.predict(y_data)

    #print(precision_value)

    svm.drawresult(x_data)