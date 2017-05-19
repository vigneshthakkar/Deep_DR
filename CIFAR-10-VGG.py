import tensorflow as tf
import os
from scipy import misc
import csv

initializer=tf.contrib.layers.xavier_initializer()

weights={
    'conv11': tf.get_variable('conv11w',[3,3,3,64],tf.float32,initializer),
    'conv12': tf.get_variable('conv12w',[3,3,64,64],tf.float32,initializer),
    'conv13': tf.get_variable('conv13w',[3,3,64,128],tf.float32,initializer),
    'conv14': tf.get_variable('conv14w',[3,3,128,128],tf.float32,initializer),

    'conv21': tf.get_variable('conv21w',[3,3,128,256],tf.float32,initializer),
    'conv22': tf.get_variable('conv22w',[3,3,256,256],tf.float32,initializer),
    'conv23': tf.get_variable('conv23w',[3,3,256,256],tf.float32,initializer),
    'conv24': tf.get_variable('conv24w',[3,3,256,256],tf.float32,initializer),

    'conv31': tf.get_variable('conv31w',[3,3,256,512],tf.float32,initializer),
    'conv32': tf.get_variable('conv32w',[3,3,512,512],tf.float32,initializer),
    'conv33': tf.get_variable('conv33w',[3,3,512,512],tf.float32,initializer),
    'conv34': tf.get_variable('conv34w',[3,3,512,512],tf.float32,initializer),

    'conv41': tf.get_variable('conv41w',[3,3,512,512],tf.float32,initializer),
    'conv42': tf.get_variable('conv42w',[3,3,512,512],tf.float32,initializer),
    'conv43': tf.get_variable('conv43w',[3,3,512,512],tf.float32,initializer),
    'conv44': tf.get_variable('conv44w',[3,3,512,512],tf.float32,initializer),

    'fc1': tf.get_variable('fc1w',[512,1024],tf.float32,initializer),
    'fc2': tf.get_variable('fc2w',[1024,1024],tf.float32,initializer),
    'fc3': tf.get_variable('fc3w',[1024,10],tf.float32,initializer),
}

biases={
    'conv11': tf.get_variable('conv11b',[64],tf.float32,initializer),
    'conv12': tf.get_variable('conv12b',[64],tf.float32,initializer),
    'conv13': tf.get_variable('conv13b',[128],tf.float32,initializer),
    'conv14': tf.get_variable('conv14b',[128],tf.float32,initializer),

    'conv21': tf.get_variable('conv21b',[256],tf.float32,initializer),
    'conv22': tf.get_variable('conv22b',[256],tf.float32,initializer),
    'conv23': tf.get_variable('conv23b',[256],tf.float32,initializer),
    'conv24': tf.get_variable('conv24b',[256],tf.float32,initializer),

    'conv31': tf.get_variable('conv31b',[512],tf.float32,initializer),
    'conv32': tf.get_variable('conv32b',[512],tf.float32,initializer),
    'conv33': tf.get_variable('conv33b',[512],tf.float32,initializer),
    'conv34': tf.get_variable('conv34b',[512],tf.float32,initializer),

    'conv41': tf.get_variable('conv41b',[512],tf.float32,initializer),
    'conv42': tf.get_variable('conv42b',[512],tf.float32,initializer),
    'conv43': tf.get_variable('conv43b',[512],tf.float32,initializer),
    'conv44': tf.get_variable('conv44b',[512],tf.float32,initializer),

    'fc1': tf.get_variable('fc1b',[1024],tf.float32,initializer),
    'fc2': tf.get_variable('fc2b',[1024],tf.float32,initializer),
    'fc3': tf.get_variable('fc3b',[10],tf.float32,initializer),
}

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

def conv(input,weight,bias,stride,name):
    with tf.name_scope(name):
        convolution=tf.add(tf.nn.conv2d(input,weight,strides=[1,stride,stride,1],padding='SAME'),bias)
        activate=tf.nn.relu(convolution)
        return activate

def maxpool(x,kernel,stride,name):
    out=tf.nn.max_pool(x,ksize=[1,kernel,kernel,1],strides=[1,stride,stride,1],padding='SAME',name=name)

def network(x):

    with tf.name_scope('Block1'):
        conv11=conv(x,weights['conv11'],biases['conv11'],1,'conv11')
        conv12=conv(conv11,weights['conv12'],biases['conv12'],1,'conv12')
        conv13=conv(conv12,weights['conv13'],biases['conv13'],1,'conv13')
        conv14=conv(conv13,weights['conv14'],biases['conv14'],1,'conv14')
        maxpool1=maxpool(conv14,2,2,'maxpool1')

    with tf.name_scope('Block2'):
        conv21=conv(maxpool1,weights['conv21'],biases['conv21'],1,'conv21')
        conv22=conv(conv21,weights['conv22'],biases['conv22'],1,'conv22')
        conv23=conv(conv22,weights['conv23'],biases['conv23'],1,'conv23')
        conv24=conv(conv23,weights['conv24'],biases['conv24'],1,'conv24')
        maxpool2=maxpool(conv24,2,2,'maxpool2')

    with tf.name_scope('Block3'):
        conv31=conv(maxpool2,weights['conv31'],biases['conv31'],1,'conv31')
        conv32=conv(conv31,weights['conv32'],biases['conv32'],1,'conv32')
        conv33=conv(conv32,weights['conv33'],biases['conv33'],1,'conv33')
        conv34=conv(conv33,weights['conv34'],biases['convw4'],1,'conv34')
        maxpool3=maxpool(conv34,2,2,'maxpool3')

    with tf.name_scope('Block4'):
        conv41=conv(maxpool3,weights['conv41'],biases['conv41'],1,'conv41')
        conv42=conv(conv41,weights['conv42'],biases['conv42'],1,'conv42')
        conv43=conv(conv42,weights['conv43'],biases['conv43'],1,'conv43')
        conv44=conv(conv43,weights['conv44'],biases['conv44'],1,'conv44')
        maxpool4=maxpool(conv44,2,2,'maxpool4')

    with tf.name_scope('fc'):
        maxpool4=tf.reshape(maxpool4,[-1,512])
        fc1=tf.add(tf.matmul(maxpool4,weights['fc1']),biases['fc1'])
        fc2=tf.add(tf.matmul(fc1,weights['fc2']),biases['fc2'])
        fc3=tf.add(tf.matmul(fc2,weights['fc3']),biases['fc3'])

    return fc3

predict_y=network(x)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
optimize=tf.train.AdamOptimizer().minimize(loss)

def save(path,sess):
    saver=tf.train.Saver()
    saver.save(sess,path)

def restore(sess):
    saver=tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint('./'))


def train():
    print('Loading train images')
    path='train/'
    count=0
    files=os.listdir(path)
    images=[]
    for file in files: count+=1; print(count); images.append(misc.imread(path+file))
    print('Images loaded')
    labels=[]
    with open('trainLabels.csv','rt') as csvfile:
        labelcsv=csv.reader(csvfile,delimiter=' ',quotechar='|')
        for row in labelcsv:
            row=row[0].rstrip()
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
        try: restore(sess)
        except: print("Couldn't restore, Reinitializing"); sess.run(tf.global_variables_initializer())
        for epoch in range(1,101):
            epochloss=0
            for i in range(1,501):
                batchimages,batchlabels=images[100*(i-1):100*(i)],labels[100*(i-1):100*i]
                batchloss,_=sess.run([loss,optimize],feed_dict={x:batchimages, y:batchlabels})
                print('Batch',i,'out of 500 completed in epoch',epoch,'. Batch loss : ', batchloss)
                epochloss+=batchloss
                if i%2==0: save('var.ckpt',sess)
            print('Epoch',epoch,'completed, loss :',epochloss)
            if epoch%2==0: save('var.ckpt',sess)
        print('Network trained')

train()
