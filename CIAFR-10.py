import tensorflow as tf
import os
from scipy import misc
import csv

initializer=tf.contrib.layers.xavier_initializer()
saver=tf.train.Saver()

weights={
    'conv1': tf.get_variable('conv1w',[3,3,3,16],tf.float32,initializer),

    'conv21': tf.get_variable('conv21w',[3,3,16,160],tf.float32,initializer),
    'conv22': tf.get_variable('conv22w',[3,3,160,160],tf.float32,initializer),
    'conv23': tf.get_variable('conv23w',[3,3,160,160],tf.float32,initializer),
    'conv24': tf.get_variable('conv24w',[3,3,160,160],tf.float32,initializer),
    'conv25': tf.get_variable('conv25w',[3,3,160,160],tf.float32,initializer),
    'conv26': tf.get_variable('conv26w',[3,3,160,160],tf.float32,initializer),
    'conv27': tf.get_variable('conv27w',[3,3,160,160],tf.float32,initializer),
    'conv28': tf.get_variable('conv28w',[3,3,160,160],tf.float32,initializer),
    'conv29': tf.get_variable('conv29w',[3,3,160,160],tf.float32,initializer),

    'conv31': tf.get_variable('conv31w',[3,3,160,320],tf.float32,initializer),
    'conv32': tf.get_variable('conv32w',[3,3,320,320],tf.float32,initializer),
    'conv33': tf.get_variable('conv33w',[3,3,320,320],tf.float32,initializer),
    'conv34': tf.get_variable('conv34w',[3,3,320,320],tf.float32,initializer),
    'conv35': tf.get_variable('conv35w',[3,3,320,320],tf.float32,initializer),
    'conv36': tf.get_variable('conv36w',[3,3,320,320],tf.float32,initializer),
    'conv37': tf.get_variable('conv37w',[3,3,320,160],tf.float32,initializer),
    'conv38': tf.get_variable('conv38w',[3,3,320,320],tf.float32,initializer),
    'conv39': tf.get_variable('conv39w',[3,3,320,320],tf.float32,initializer),

    'conv41': tf.get_variable('conv41w',[3,3,320,640],tf.float32,initializer),
    'conv42': tf.get_variable('conv42w',[3,3,640,640],tf.float32,initializer),
    'conv43': tf.get_variable('conv43w',[3,3,640,640],tf.float32,initializer),
    'conv44': tf.get_variable('conv44w',[3,3,640,640],tf.float32,initializer),
    'conv45': tf.get_variable('conv45w',[3,3,640,640],tf.float32,initializer),
    'conv46': tf.get_variable('conv46w',[3,3,640,640],tf.float32,initializer),
    'conv47': tf.get_variable('conv47w',[3,3,640,640],tf.float32,initializer),
    'conv48': tf.get_variable('conv48w',[3,3,640,640],tf.float32,initializer),
    'conv49': tf.get_variable('conv49w',[3,3,640,640],tf.float32,initializer),

    'fc': tf.get_variable('fcw',[640,10],tf.float32,initializer)
}

biases={
    'conv1': tf.get_variable('conv1b',[16],tf.float32,initializer),

    'conv21': tf.get_variable('conv21b',[160],tf.float32,initializer),
    'conv22': tf.get_variable('conv22b',[160],tf.float32,initializer),
    'conv23': tf.get_variable('conv23b',[160],tf.float32,initializer),
    'conv24': tf.get_variable('conv24b',[160],tf.float32,initializer),
    'conv25': tf.get_variable('conv25b',[160],tf.float32,initializer),
    'conv26': tf.get_variable('conv26b',[160],tf.float32,initializer),
    'conv27': tf.get_variable('conv27b',[160],tf.float32,initializer),
    'conv28': tf.get_variable('conv28b',[160],tf.float32,initializer),
    'conv29': tf.get_variable('conv29b',[160],tf.float32,initializer),

    'conv31': tf.get_variable('conv31b',[320],tf.float32,initializer),
    'conv32': tf.get_variable('conv32b',[320],tf.float32,initializer),
    'conv33': tf.get_variable('conv33b',[320],tf.float32,initializer),
    'conv34': tf.get_variable('conv34b',[320],tf.float32,initializer),
    'conv35': tf.get_variable('conv35b',[320],tf.float32,initializer),
    'conv36': tf.get_variable('conv36b',[320],tf.float32,initializer),
    'conv37': tf.get_variable('conv37b',[320],tf.float32,initializer),
    'conv38': tf.get_variable('conv38b',[320],tf.float32,initializer),
    'conv39': tf.get_variable('conv39b',[320],tf.float32,initializer),

    'conv41': tf.get_variable('conv41b',[640],tf.float32,initializer),
    'conv42': tf.get_variable('conv42b',[640],tf.float32,initializer),
    'conv43': tf.get_variable('conv43b',[640],tf.float32,initializer),
    'conv44': tf.get_variable('conv44b',[640],tf.float32,initializer),
    'conv45': tf.get_variable('conv45b',[640],tf.float32,initializer),
    'conv46': tf.get_variable('conv46b',[640],tf.float32,initializer),
    'conv47': tf.get_variable('conv47b',[640],tf.float32,initializer),
    'conv48': tf.get_variable('conv48b',[640],tf.float32,initializer),
    'conv49': tf.get_variable('conv49b',[640],tf.float32,initializer),

    'fc': tf.get_variable('fcb',[10],tf.float32,initializer)
}

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)
istraining=tf.placeholder(tf.bool)

def conv(input,weight,bias,stride,istraining,name):
    with tf.name_scope(name):
        batchnorm=tf.contrib.layers.batch_norm(input,center=True,Scale=True,is_training=istraining)
        activate=tf.nn.relu(batchnorm)
        convolution=tf.add(tf.nn.conv2d(activate,weight,strides=[1,stride,stride,1],padding='SAME'),bias)
        return convolution

def network(x,istraining):

    with tf.name_scope('Block 1'):
        conv1=tf.add(tf.nn.conv2d(x,weights['conv1'],strides=[1,1,1,1], padding='SAME')+biases['conv1'],name='conv1')

    with tf.name_scope('Block 2'):
        conv21=conv(conv1,weights['conv21'],biases['conv21'],1,istraining,'conv21')
        conv22=conv(conv21,weights['conv22'],biases['conv22'],1,istraining,'conv22')
        conv23=tf.add(conv(conv22,weights['conv23'],biases['conv23'],1,istraining,'conv23')+conv21)
        conv24=conv(conv23,weights['conv24'],biases['conv24'],1,istraining,'conv24')
        conv25=tf.add(conv(conv24,weights['conv25'],biases['conv25'],1,istraining,'conv25')+conv23)
        conv26=conv(conv25,weights['conv26'],biases['conv26'],1,istraining,'conv26')
        conv27=tf.add(conv(conv26,weights['conv27'],biases['conv27'],1,istraining,'conv27')+conv25)
        conv28=conv(conv27,weights['conv28'],biases['conv28'],1,istraining,'conv28')
        conv29=tf.add(conv(conv28,weights['conv29'],biases['conv29'],1,istraining,'conv29')+conv27)

    with tf.name_scope('Block 3'):
        conv31=conv(conv29,weights['conv31'],biases['conv31'],2,istraining,'conv31')
        conv32=conv(conv31,weights['conv32'],biases['conv32'],1,istraining,'conv32')
        conv33=tf.add(conv(conv32,weights['conv33'],biases['conv33'],1,istraining,'conv33')+conv31)
        conv34=conv(conv33,weights['conv34'],biases['conv34'],1,istraining,'conv34')
        conv35=tf.add(conv(conv34,weights['conv35'],biases['conv35'],1,istraining,'conv35')+conv33)
        conv36=conv(conv35,weights['conv36'],biases['conv36'],1,istraining,'conv36')
        conv37=tf.add(conv(conv36,weights['conv37'],biases['conv37'],1,istraining,'conv37')+conv35)
        conv38=conv(conv37,weights['conv38'],biases['conv38'],1,istraining,'conv38')
        conv39=tf.add(conv(conv38,weights['conv39'],biases['conv39'],1,istraining,'conv39')+conv37)

    with tf.name_scope('Block 4'):
        conv41=conv(conv39,weights['conv41'],biases['conv41'],2,istraining,'conv41')
        conv42=conv(conv41,weights['conv42'],biases['conv42'],1,istraining,'conv42')
        conv43=tf.add(conv(conv42,weights['conv43'],biases['conv43'],1,istraining,'conv43')+conv31)
        conv44=conv(conv43,weights['conv44'],biases['conv44'],1,istraining,'conv44')
        conv45=tf.add(conv(conv44,weights['conv45'],biases['conv45'],1,istraining,'conv45')+conv43)
        conv46=conv(conv45,weights['conv46'],biases['conv46'],1,istraining,'conv46')
        conv47=tf.add(conv(conv46,weights['conv47'],biases['conv47'],1,istraining,'conv47')+conv45)
        conv48=conv(conv47,weights['conv48'],biases['conv48'],1,istraining,'conv48')
        conv49=tf.add(conv(conv48,weights['conv49'],biases['conv49'],1,istraining,'conv49')+conv47)

    with tf.name_scope('Avg Pool'):
        avgpool=tf.nn.avg_pool(conv49,ksize=[1,8,8,1],strides=[1,1,1,1],padding='VALID')

    with tf.name_scope('fc'):
        avgpool=tf.reshape(avgpool,[-1,1*1*640])
        fc=tf.add(tf.matmul(avgpool,weights['fc']),biases['fc'])

    return fc

predict_y=network(x,istraining)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
optimize=tf.train.AdamOptimizer().minimize(loss)

def save(path,sess):
    saver.save(sess,path)

def restore(path,sess):
    saver.restore(sess,path)


def train():
    path='/train/'
    files=os.listdir(path)
    images=[]
    for file in files: images.append(misc.imread(path+file))
    print('Images loaded')
    labels=[]
    with open('trainLabels.csv','rb') as csvfile:
        labelcsv=csv.reader(csvfile,delimiter=' ',quotechar='|')
        for row in labelcsv:
            row=row.rstrip()
            row=row.split(',')
            row=row[-1]
            if row=='airplane': labels.append([1,0,0,0,0,0,0,0,0,0])
            if row=='automobile': labels.append([0,1,0,0,0,0,0,0,0,0])
            if row=='bird': labels.append([0,0,1,0,0,0,0,0,0,0])
            if row=='cat': labels.append([0,0,0,1,0,0,0,0,0,0])
            if row=='deer': labels.append([0,0,0,0,1,0,0,0,0,0])
            if row=='dog': labels.append([0,0,0,0,0,1,0,0,0,0])
            if row=='frog': labels.append([0,0,0,0,0,0,1,0,0,0])
            if row=='horse': labels.append([0,0,0,0,0,0,0,1,0,0])
            if row=='ship': labels.append([0,0,0,0,0,0,0,0,1,0])
            if row=='truck': labels.append([0,0,0,0,0,0,0,0,0,1])
    print('Labels loaded')
    print('Batch size : 100')
    print('Variables saved after every 100 Epochs')
    print('1000 Epochs')
    with tf.Session() as sess:
        for epoch in range(1,1001):
            epochloss=0
            for i in range(1,501):
                batchimages,batchlabels=images[100*(i-1):100*(i)],labels[100*(i-1):100*i]
                loss,optimize=sess.run([loss,optimize],feed_dict={x:batchimages, y:batchlabels, istraining:True})
                epochloss+=loss
            print('Epoch',epoch,'completed, loss :',epochloss)
            if epoch%100==0:
                save('/Variables/var'+str(epoch)+'.ckpt',sess)
                print('Variables saved')
        print('Network trained')

train()
