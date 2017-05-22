import tensorflow as tf
import os
from scipy import misc
import csv

initializer=tf.contrib.layers.xavier_initializer()

weights={
    'conv1': tf.get_variable('conv1', [7,7,3,16],tf.float32,initializer),

    'd1conv11': tf.get_variable('d1conv11w',[1,1,16,64],tf.float32,initializer),
    'd133conv1': tf.get_variable('d1conv12w',[3,3,64,16],tf.float32,initializer),
    'd1conv21': tf.get_variable('d1conv21w',[1,1,32,64],tf.float32,initializer),
    'd1conv31': tf.get_variable('d1conv31w',[1,1,48,64],tf.float32,initializer),
    'd1conv41': tf.get_variable('d1conv41w',[1,1,64,64],tf.float32,initializer),
    'd133conv2': tf.get_variable('d1conv42w',[3,3,64,16],tf.float32,initializer),
    'd1conv51': tf.get_variable('d1conv51w',[1,1,80,64],tf.float32,initializer),
    'd1conv61': tf.get_variable('d1conv61w',[1,1,96,64],tf.float32,initializer),

    't1conv': tf.get_variable('t1convw',[1,1,112,56],tf.float32,initializer),

    'd2conv11': tf.get_variable('d2conv11w',[1,1,56,64],tf.float32,initializer),
    'd233conv1': tf.get_variable('d2conv12w',[3,3,64,16],tf.float32,initializer),
    'd2conv21': tf.get_variable('d2conv21w',[1,1,72,64],tf.float32,initializer),
    'd2conv31': tf.get_variable('d2conv31w',[1,1,88,64],tf.float32,initializer),
    'd2conv41': tf.get_variable('d2conv41w',[1,1,104,64],tf.float32,initializer),
    'd2conv51': tf.get_variable('d2conv51w',[1,1,120,64],tf.float32,initializer),
    'd2conv61': tf.get_variable('d2conv61w',[1,1,136,64],tf.float32,initializer),
    'd2conv71': tf.get_variable('d2conv71w',[1,1,152,64],tf.float32,initializer),
    'd233conv2': tf.get_variable('d2conv72w',[3,3,64,16],tf.float32,initializer),
    'd2conv81': tf.get_variable('d2conv81w',[1,1,168,64],tf.float32,initializer),
    'd2conv91': tf.get_variable('d2conv91w',[1,1,184,64],tf.float32,initializer),
    'd2conv101': tf.get_variable('d2conv101w',[1,1,200,64],tf.float32,initializer),
    'd2conv111': tf.get_variable('d2conv111w',[1,1,216,64],tf.float32,initializer),
    'd2conv121': tf.get_variable('d2conv121w',[1,1,232,64],tf.float32,initializer),

    't2conv': tf.get_variable('t2convw',[1,1,248,124],tf.float32,initializer),

    'd3conv11': tf.get_variable('d3conv11w',[1,1,124,64],tf.float32,initializer),
    'd333conv1': tf.get_variable('d3conv12w',[3,3,64,16],tf.float32,initializer),
    'd3conv21': tf.get_variable('d3conv21w',[1,1,140,64],tf.float32,initializer),
    'd3conv31': tf.get_variable('d3conv31w',[1,1,156,64],tf.float32,initializer),
    'd3conv41': tf.get_variable('d3conv41w',[1,1,172,64],tf.float32,initializer),
    'd3conv51': tf.get_variable('d3conv51w',[1,1,188,64],tf.float32,initializer),
    'd3conv61': tf.get_variable('d3conv61w',[1,1,204,64],tf.float32,initializer),
    'd3conv71': tf.get_variable('d3conv71w',[1,1,220,64],tf.float32,initializer),
    'd3conv81': tf.get_variable('d3conv81w',[1,1,236,64],tf.float32,initializer),
    'd3conv91': tf.get_variable('d3conv91w',[1,1,252,64],tf.float32,initializer),
    'd3conv101': tf.get_variable('d3conv101w',[1,1,268,64],tf.float32,initializer),
    'd3conv111': tf.get_variable('d3conv111w',[1,1,284,64],tf.float32,initializer),
    'd3conv121': tf.get_variable('d3conv121w',[1,1,300,64],tf.float32,initializer),
    'd3conv131': tf.get_variable('d3conv131w',[1,1,316,64],tf.float32,initializer),
    'd3conv141': tf.get_variable('d3conv141w',[1,1,332,64],tf.float32,initializer),
    'd3conv151': tf.get_variable('d3conv151w',[1,1,348,64],tf.float32,initializer),
    'd3conv161': tf.get_variable('d3conv161w',[1,1,364,64],tf.float32,initializer),
    'd3conv171': tf.get_variable('d3conv171w',[1,1,380,64],tf.float32,initializer),
    'd3conv181': tf.get_variable('d3conv181w',[1,1,396,64],tf.float32,initializer),
    'd3conv191': tf.get_variable('d3conv191w',[1,1,412,64],tf.float32,initializer),
    'd3conv201': tf.get_variable('d3conv201w',[1,1,428,64],tf.float32,initializer),
    'd3conv211': tf.get_variable('d3conv211w',[1,1,444,64],tf.float32,initializer),
    'd3conv221': tf.get_variable('d3conv221w',[1,1,460,64],tf.float32,initializer),
    'd3conv231': tf.get_variable('d3conv231w',[1,1,476,64],tf.float32,initializer),
    'd3conv241': tf.get_variable('d3conv241w',[1,1,492,64],tf.float32,initializer),
    'd3conv251': tf.get_variable('d3conv251w',[1,1,508,64],tf.float32,initializer),
    'd333conv2': tf.get_variable('d3conv252w',[3,3,64,16],tf.float32,initializer),
    'd3conv261': tf.get_variable('d3conv261w',[1,1,524,64],tf.float32,initializer),
    'd3conv271': tf.get_variable('d3conv271w',[1,1,540,64],tf.float32,initializer),
    'd3conv281': tf.get_variable('d3conv281w',[1,1,556,64],tf.float32,initializer),
    'd3conv291': tf.get_variable('d3conv291w',[1,1,572,64],tf.float32,initializer),
    'd3conv301': tf.get_variable('d3conv301w',[1,1,588,64],tf.float32,initializer),
    'd3conv311': tf.get_variable('d3conv311w',[1,1,604,64],tf.float32,initializer),
    'd3conv321': tf.get_variable('d3conv321w',[1,1,620,64],tf.float32,initializer),
    'd3conv331': tf.get_variable('d3conv331w',[1,1,636,64],tf.float32,initializer),
    'd3conv341': tf.get_variable('d3conv341w',[1,1,652,64],tf.float32,initializer),
    'd3conv351': tf.get_variable('d3conv351w',[1,1,668,64],tf.float32,initializer),
    'd3conv361': tf.get_variable('d3conv361w',[1,1,684,64],tf.float32,initializer),
    'd3conv371': tf.get_variable('d3conv371w',[1,1,700,64],tf.float32,initializer),
    'd3conv381': tf.get_variable('d3conv381w',[1,1,716,64],tf.float32,initializer),
    'd3conv391': tf.get_variable('d3conv391w',[1,1,732,64],tf.float32,initializer),
    'd3conv401': tf.get_variable('d3conv401w',[1,1,748,64],tf.float32,initializer),
    'd3conv411': tf.get_variable('d3conv411w',[1,1,764,64],tf.float32,initializer),
    'd3conv421': tf.get_variable('d3conv421w',[1,1,780,64],tf.float32,initializer),
    'd3conv431': tf.get_variable('d3conv431w',[1,1,796,64],tf.float32,initializer),
    'd3conv441': tf.get_variable('d3conv441w',[1,1,812,64],tf.float32,initializer),
    'd3conv451': tf.get_variable('d3conv451w',[1,1,828,64],tf.float32,initializer),
    'd3conv461': tf.get_variable('d3conv461w',[1,1,844,64],tf.float32,initializer),
    'd3conv471': tf.get_variable('d3conv471w',[1,1,860,64],tf.float32,initializer),
    'd3conv481': tf.get_variable('d3conv481w',[1,1,876,64],tf.float32,initializer),

    't3conv': tf.get_variable('t3convw',[1,1,892,446],tf.float32,initializer),

    'd4conv11': tf.get_variable('d4conv11w',[1,1,446,64],tf.float32,initializer),
    'd433conv1': tf.get_variable('d4conv12w',[3,3,64,16],tf.float32,initializer),
    'd4conv21': tf.get_variable('d4conv21w',[1,1,462,64],tf.float32,initializer),
    'd4conv31': tf.get_variable('d4conv31w',[1,1,478,64],tf.float32,initializer),
    'd4conv41': tf.get_variable('d4conv41w',[1,1,494,64],tf.float32,initializer),
    'd4conv51': tf.get_variable('d4conv51w',[1,1,510,64],tf.float32,initializer),
    'd4conv61': tf.get_variable('d4conv61w',[1,1,526,64],tf.float32,initializer),
    'd4conv71': tf.get_variable('d4conv71w',[1,1,542,64],tf.float32,initializer),
    'd4conv81': tf.get_variable('d4conv81w',[1,1,558,64],tf.float32,initializer),
    'd4conv91': tf.get_variable('d4conv91w',[1,1,574,64],tf.float32,initializer),
    'd4conv101': tf.get_variable('d4conv101w',[1,1,590,64],tf.float32,initializer),
    'd4conv111': tf.get_variable('d4conv111w',[1,1,606,64],tf.float32,initializer),
    'd4conv121': tf.get_variable('d4conv121w',[1,1,622,64],tf.float32,initializer),
    'd4conv131': tf.get_variable('d4conv131w',[1,1,638,64],tf.float32,initializer),
    'd4conv141': tf.get_variable('d4conv141w',[1,1,654,64],tf.float32,initializer),
    'd4conv151': tf.get_variable('d4conv151w',[1,1,670,64],tf.float32,initializer),
    'd4conv161': tf.get_variable('d4conv161w',[1,1,686,64],tf.float32,initializer),
    'd4conv171': tf.get_variable('d4conv171w',[1,1,702,64],tf.float32,initializer),
    'd433conv2': tf.get_variable('d4conv172w',[3,3,64,16],tf.float32,initializer),
    'd4conv181': tf.get_variable('d4conv181w',[1,1,718,64],tf.float32,initializer),
    'd4conv191': tf.get_variable('d4conv191w',[1,1,734,64],tf.float32,initializer),
    'd4conv201': tf.get_variable('d4conv201w',[1,1,750,64],tf.float32,initializer),
    'd4conv211': tf.get_variable('d4conv211w',[1,1,766,64],tf.float32,initializer),
    'd4conv221': tf.get_variable('d4conv221w',[1,1,782,64],tf.float32,initializer),
    'd4conv231': tf.get_variable('d4conv231w',[1,1,798,64],tf.float32,initializer),
    'd4conv241': tf.get_variable('d4conv241w',[1,1,814,64],tf.float32,initializer),
    'd4conv251': tf.get_variable('d4conv251w',[1,1,830,64],tf.float32,initializer),
    'd4conv261': tf.get_variable('d4conv261w',[1,1,846,64],tf.float32,initializer),
    'd4conv271': tf.get_variable('d4conv271w',[1,1,862,64],tf.float32,initializer),
    'd4conv281': tf.get_variable('d4conv281w',[1,1,878,64],tf.float32,initializer),
    'd4conv291': tf.get_variable('d4conv291w',[1,1,894,64],tf.float32,initializer),
    'd4conv301': tf.get_variable('d4conv301w',[1,1,910,64],tf.float32,initializer),
    'd4conv311': tf.get_variable('d4conv311w',[1,1,926,64],tf.float32,initializer),
    'd4conv321': tf.get_variable('d4conv321w',[1,1,942,64],tf.float32,initializer),

    'fc': tf.get_variable('fcw',[958,10],tf.float32,initializer),
}

biases={
    'conv1': tf.get_variable('conv1b',[16],tf.float32,initializer),

    'd1conv11': tf.get_variable('d1conv11b',[64],tf.float32,initializer),
    'd133conv1': tf.get_variable('d1conv12b',[16],tf.float32,initializer),
    'd1conv21': tf.get_variable('d1conv21b',[64],tf.float32,initializer),
    'd1conv31': tf.get_variable('d1conv31b',[64],tf.float32,initializer),
    'd1conv41': tf.get_variable('d1conv41b',[64],tf.float32,initializer),
    'd133conv2': tf.get_variable('d1conv42b',[16],tf.float32,initializer),
    'd1conv51': tf.get_variable('d1conv51b',[64],tf.float32,initializer),
    'd1conv61': tf.get_variable('d1conv61b',[64],tf.float32,initializer),

    't1conv': tf.get_variable('t1convb',[56],tf.float32,initializer),

    'd2conv11': tf.get_variable('d2conv11b',[64],tf.float32,initializer),
    'd233conv1': tf.get_variable('d2conv12b',[16],tf.float32,initializer),
    'd2conv21': tf.get_variable('d2conv21b',[64],tf.float32,initializer),
    'd2conv31': tf.get_variable('d2conv31b',[64],tf.float32,initializer),
    'd2conv41': tf.get_variable('d2conv41b',[64],tf.float32,initializer),
    'd2conv51': tf.get_variable('d2conv51b',[64],tf.float32,initializer),
    'd2conv61': tf.get_variable('d2conv61b',[64],tf.float32,initializer),
    'd2conv71': tf.get_variable('d2conv71b',[64],tf.float32,initializer),
    'd233conv2': tf.get_variable('d2conv72b',[16],tf.float32,initializer),
    'd2conv81': tf.get_variable('d2conv81b',[64],tf.float32,initializer),
    'd2conv91': tf.get_variable('d2conv91b',[64],tf.float32,initializer),
    'd2conv101': tf.get_variable('d2conv101b',[64],tf.float32,initializer),
    'd2conv111': tf.get_variable('d2conv111b',[64],tf.float32,initializer),
    'd2conv121': tf.get_variable('d2conv121b',[64],tf.float32,initializer),

    't2conv': tf.get_variable('t2conv',[124],tf.float32,initializer),

    'd3conv11': tf.get_variable('d3conv11b',[64],tf.float32,initializer),
    'd333conv1': tf.get_variable('d3conv12b',[16],tf.float32,initializer),
    'd3conv21': tf.get_variable('d3conv21b',[64],tf.float32,initializer),
    'd3conv31': tf.get_variable('d3conv31b',[64],tf.float32,initializer),
    'd3conv41': tf.get_variable('d3conv41b',[64],tf.float32,initializer),
    'd3conv51': tf.get_variable('d3conv51b',[64],tf.float32,initializer),
    'd3conv61': tf.get_variable('d3conv61b',[64],tf.float32,initializer),
    'd3conv71': tf.get_variable('d3conv71b',[64],tf.float32,initializer),
    'd3conv81': tf.get_variable('d3conv81b',[64],tf.float32,initializer),
    'd3conv91': tf.get_variable('d3conv91b',[64],tf.float32,initializer),
    'd3conv101': tf.get_variable('d3conv101b',[64],tf.float32,initializer),
    'd3conv111': tf.get_variable('d3conv111b',[64],tf.float32,initializer),
    'd3conv121': tf.get_variable('d3conv121b',[64],tf.float32,initializer),
    'd3conv131': tf.get_variable('d3conv131b',[64],tf.float32,initializer),
    'd3conv141': tf.get_variable('d3conv141b',[64],tf.float32,initializer),
    'd3conv151': tf.get_variable('d3conv151b',[64],tf.float32,initializer),
    'd3conv161': tf.get_variable('d3conv161b',[64],tf.float32,initializer),
    'd3conv171': tf.get_variable('d3conv171b',[64],tf.float32,initializer),
    'd3conv181': tf.get_variable('d3conv181b',[64],tf.float32,initializer),
    'd3conv191': tf.get_variable('d3conv191b',[64],tf.float32,initializer),
    'd3conv201': tf.get_variable('d3conv201b',[64],tf.float32,initializer),
    'd3conv211': tf.get_variable('d3conv211b',[64],tf.float32,initializer),
    'd3conv221': tf.get_variable('d3conv221b',[64],tf.float32,initializer),
    'd3conv231': tf.get_variable('d3conv231b',[64],tf.float32,initializer),
    'd3conv241': tf.get_variable('d3conv241b',[64],tf.float32,initializer),
    'd3conv251': tf.get_variable('d3conv251b',[64],tf.float32,initializer),
    'd333conv2': tf.get_variable('d3conv252b',[16],tf.float32,initializer),
    'd3conv261': tf.get_variable('d3conv261b',[64],tf.float32,initializer),
    'd3conv271': tf.get_variable('d3conv271b',[64],tf.float32,initializer),
    'd3conv281': tf.get_variable('d3conv281b',[64],tf.float32,initializer),
    'd3conv291': tf.get_variable('d3conv291b',[64],tf.float32,initializer),
    'd3conv301': tf.get_variable('d3conv301b',[64],tf.float32,initializer),
    'd3conv311': tf.get_variable('d3conv311b',[64],tf.float32,initializer),
    'd3conv321': tf.get_variable('d3conv321b',[64],tf.float32,initializer),
    'd3conv331': tf.get_variable('d3conv331b',[64],tf.float32,initializer),
    'd3conv341': tf.get_variable('d3conv341b',[64],tf.float32,initializer),
    'd3conv351': tf.get_variable('d3conv351b',[64],tf.float32,initializer),
    'd3conv361': tf.get_variable('d3conv361b',[64],tf.float32,initializer),
    'd3conv371': tf.get_variable('d3conv371b',[64],tf.float32,initializer),
    'd3conv381': tf.get_variable('d3conv381b',[64],tf.float32,initializer),
    'd3conv391': tf.get_variable('d3conv391b',[64],tf.float32,initializer),
    'd3conv401': tf.get_variable('d3conv401b',[64],tf.float32,initializer),
    'd3conv411': tf.get_variable('d3conv411b',[64],tf.float32,initializer),
    'd3conv421': tf.get_variable('d3conv421b',[64],tf.float32,initializer),
    'd3conv431': tf.get_variable('d3conv431b',[64],tf.float32,initializer),
    'd3conv441': tf.get_variable('d3conv441b',[64],tf.float32,initializer),
    'd3conv451': tf.get_variable('d3conv451b',[64],tf.float32,initializer),
    'd3conv461': tf.get_variable('d3conv461b',[64],tf.float32,initializer),
    'd3conv471': tf.get_variable('d3conv471b',[64],tf.float32,initializer),
    'd3conv481': tf.get_variable('d3conv481b',[64],tf.float32,initializer),

    't3conv': tf.get_variable('t3conv',[446],tf.float32,initializer),

    'd4conv11': tf.get_variable('d4conv11b',[64],tf.float32,initializer),
    'd433conv1': tf.get_variable('d4conv12b',[16],tf.float32,initializer),
    'd4conv21': tf.get_variable('d4conv21b',[64],tf.float32,initializer),
    'd4conv31': tf.get_variable('d4conv31b',[64],tf.float32,initializer),
    'd4conv41': tf.get_variable('d4conv41b',[64],tf.float32,initializer),
    'd4conv51': tf.get_variable('d4conv51b',[64],tf.float32,initializer),
    'd4conv61': tf.get_variable('d4conv61b',[64],tf.float32,initializer),
    'd4conv71': tf.get_variable('d4conv71b',[64],tf.float32,initializer),
    'd4conv81': tf.get_variable('d4conv81b',[64],tf.float32,initializer),
    'd4conv91': tf.get_variable('d4conv91b',[64],tf.float32,initializer),
    'd4conv101': tf.get_variable('d4conv101b',[64],tf.float32,initializer),
    'd4conv111': tf.get_variable('d4conv111b',[64],tf.float32,initializer),
    'd4conv121': tf.get_variable('d4conv121b',[64],tf.float32,initializer),
    'd4conv131': tf.get_variable('d4conv131b',[64],tf.float32,initializer),
    'd4conv141': tf.get_variable('d4conv141b',[64],tf.float32,initializer),
    'd4conv151': tf.get_variable('d4conv151b',[64],tf.float32,initializer),
    'd4conv161': tf.get_variable('d4conv161b',[64],tf.float32,initializer),
    'd4conv171': tf.get_variable('d4conv171b',[64],tf.float32,initializer),
    'd433conv2': tf.get_variable('d4conv172b',[16],tf.float32,initializer),
    'd4conv181': tf.get_variable('d4conv181b',[64],tf.float32,initializer),
    'd4conv191': tf.get_variable('d4conv191b',[64],tf.float32,initializer),
    'd4conv201': tf.get_variable('d4conv201b',[64],tf.float32,initializer),
    'd4conv211': tf.get_variable('d4conv211b',[64],tf.float32,initializer),
    'd4conv221': tf.get_variable('d4conv221b',[64],tf.float32,initializer),
    'd4conv231': tf.get_variable('d4conv231b',[64],tf.float32,initializer),
    'd4conv241': tf.get_variable('d4conv241b',[64],tf.float32,initializer),
    'd4conv251': tf.get_variable('d4conv251b',[64],tf.float32,initializer),
    'd4conv261': tf.get_variable('d4conv261b',[64],tf.float32,initializer),
    'd4conv271': tf.get_variable('d4conv271b',[64],tf.float32,initializer),
    'd4conv281': tf.get_variable('d4conv281b',[64],tf.float32,initializer),
    'd4conv291': tf.get_variable('d4conv291b',[64],tf.float32,initializer),
    'd4conv301': tf.get_variable('d4conv301b',[64],tf.float32,initializer),
    'd4conv311': tf.get_variable('d4conv311b',[64],tf.float32,initializer),
    'd4conv321': tf.get_variable('d4conv321b',[64],tf.float32,initializer),

    'fc': tf.get_variable('fcb',[10],tf.float32,initializer),
}

x=tf.placeholder(tf.float32,[100,32,32,3])
y=tf.placeholder(tf.float32)

def conv(x,weight,bias):
    batchnorm=tf.layers.batch_normalization(x)
    activate=tf.nn.relu(batchnorm,name='relu')
    convolution=tf.add(tf.nn.conv2d(activate,weight,strides=[1,1,1,1],padding='SAME'),bias,name='conv')
    return convolution

def subblock(x,weight11,bias11,weight33,bias33):
    conv11=conv(x,weight11,bias11)
    conv33=conv(conv11,weight33,bias33)
    return tf.concat([x,conv33],axis=3)

def avgpool(x,kernel,stride):
    return tf.nn.avg_pool(x,ksize=[1,kernel,kernel,1],strides=[1,stride,stride,1],padding='SAME')

def maxpool(x,kernel,stride):
    return tf.nn.max_pool(x,ksize=[1,kernel,kernel,1],strides=[1,stride,stride,1],padding='SAME')

def transition(x,weight11,bias11):
    conv11=conv(x,weight11,bias11)
    return avgpool(conv11,2,2)

def classification(x,weightfc,biasfc):
    globalpool=tf.reshape(tf.nn.avg_pool(x,ksize=[1,4,4,1],strides=[1,1,1,1],padding='VALID'),[-1,958])
    fc=tf.add(tf.matmul(globalpool,weightfc),biasfc)
    return fc

def network(x):
    with tf.name_scope('InputLayer'):
        conv1=conv(x,weights['conv1'],biases['conv1'])
        pool=maxpool(conv1,3,1)

    with tf.name_scope('DenseBock1'):
        d1conv1=subblock(pool,weights['d1conv11'],biases['d1conv11'],weights['d133conv1'],biases['d133conv1'])
        d1conv2=subblock(d1conv1,weights['d1conv21'],biases['d1conv21'],weights['d133conv1'],biases['d133conv1'])
        d1conv3=subblock(d1conv2,weights['d1conv31'],biases['d1conv31'],weights['d133conv1'],biases['d133conv1'])
        d1conv4=subblock(d1conv3,weights['d1conv41'],biases['d1conv41'],weights['d133conv2'],biases['d133conv2'])
        d1conv5=subblock(d1conv4,weights['d1conv51'],biases['d1conv51'],weights['d133conv2'],biases['d133conv2'])
        d1conv6=subblock(d1conv5,weights['d1conv61'],biases['d1conv61'],weights['d133conv2'],biases['d133conv2'])

    with tf.name_scope('Transition1'):
        t1=transition(d1conv6,weights['t1conv'],biases['t1conv'])

    with tf.name_scope('DenseBlock2'):
        d2conv1=subblock(t1,weights['d2conv11'],biases['d2conv11'],weights['d233conv1'],biases['d233conv1'])
        d2conv2=subblock(d2conv1,weights['d2conv21'],biases['d2conv21'],weights['d233conv1'],biases['d233conv1'])
        d2conv3=subblock(d2conv2,weights['d2conv31'],biases['d2conv31'],weights['d233conv1'],biases['d233conv1'])
        d2conv4=subblock(d2conv3,weights['d2conv41'],biases['d2conv41'],weights['d233conv1'],biases['d233conv1'])
        d2conv5=subblock(d2conv4,weights['d2conv51'],biases['d2conv51'],weights['d233conv1'],biases['d233conv1'])
        d2conv6=subblock(d2conv5,weights['d2conv61'],biases['d2conv61'],weights['d233conv1'],biases['d233conv1'])

        d2conv7=subblock(d2conv6,weights['d2conv71'],biases['d2conv71'],weights['d233conv2'],biases['d233conv2'])
        d2conv8=subblock(d2conv7,weights['d2conv81'],biases['d2conv81'],weights['d233conv2'],biases['d233conv2'])
        d2conv9=subblock(d2conv8,weights['d2conv91'],biases['d2conv91'],weights['d233conv2'],biases['d233conv2'])
        d2conv10=subblock(d2conv9,weights['d2conv101'],biases['d2conv101'],weights['d233conv2'],biases['d233conv2'])
        d2conv11=subblock(d2conv10,weights['d2conv111'],biases['d2conv111'],weights['d233conv2'],biases['d233conv2'])
        d2conv12=subblock(d2conv11,weights['d2conv121'],biases['d2conv121'],weights['d233conv2'],biases['d233conv2'])

    with tf.name_scope('Transition2'):
        t2=transition(d2conv12,weights['t2conv'],biases['t2conv'])

    with tf.name_scope('DenseBlock3'):
        d3conv1=subblock(t2,weights['d3conv11'],biases['d3conv11'],weights['d333conv1'],biases['d333conv1'])
        d3conv2=subblock(d3conv1,weights['d3conv21'],biases['d3conv21'],weights['d333conv1'],biases['d333conv1'])
        d3conv3=subblock(d3conv2,weights['d3conv31'],biases['d3conv31'],weights['d333conv1'],biases['d333conv1'])
        d3conv4=subblock(d3conv3,weights['d3conv41'],biases['d3conv41'],weights['d333conv1'],biases['d333conv1'])
        d3conv5=subblock(d3conv4,weights['d3conv51'],biases['d3conv51'],weights['d333conv1'],biases['d333conv1'])
        d3conv6=subblock(d3conv5,weights['d3conv61'],biases['d3conv61'],weights['d333conv1'],biases['d333conv1'])
        d3conv7=subblock(d3conv6,weights['d3conv71'],biases['d3conv71'],weights['d333conv1'],biases['d333conv1'])
        d3conv8=subblock(d3conv7,weights['d3conv81'],biases['d3conv81'],weights['d333conv1'],biases['d333conv1'])
        d3conv9=subblock(d3conv8,weights['d3conv91'],biases['d3conv91'],weights['d333conv1'],biases['d333conv1'])
        d3conv10=subblock(d3conv9,weights['d3conv101'],biases['d3conv101'],weights['d333conv1'],biases['d333conv1'])
        d3conv11=subblock(d3conv10,weights['d3conv111'],biases['d3conv111'],weights['d333conv1'],biases['d333conv1'])
        d3conv12=subblock(d3conv11,weights['d3conv121'],biases['d3conv121'],weights['d333conv1'],biases['d333conv1'])
        d3conv13=subblock(d3conv12,weights['d3conv131'],biases['d3conv131'],weights['d333conv1'],biases['d333conv1'])
        d3conv14=subblock(d3conv13,weights['d3conv141'],biases['d3conv141'],weights['d333conv1'],biases['d333conv1'])
        d3conv15=subblock(d3conv14,weights['d3conv151'],biases['d3conv151'],weights['d333conv1'],biases['d333conv1'])
        d3conv16=subblock(d3conv15,weights['d3conv161'],biases['d3conv161'],weights['d333conv1'],biases['d333conv1'])
        d3conv17=subblock(d3conv16,weights['d3conv171'],biases['d3conv171'],weights['d333conv1'],biases['d333conv1'])
        d3conv18=subblock(d3conv17,weights['d3conv181'],biases['d3conv181'],weights['d333conv1'],biases['d333conv1'])
        d3conv19=subblock(d3conv18,weights['d3conv191'],biases['d3conv191'],weights['d333conv1'],biases['d333conv1'])
        d3conv20=subblock(d3conv19,weights['d3conv201'],biases['d3conv201'],weights['d333conv1'],biases['d333conv1'])
        d3conv21=subblock(d3conv20,weights['d3conv211'],biases['d3conv211'],weights['d333conv1'],biases['d333conv1'])
        d3conv22=subblock(d3conv21,weights['d3conv221'],biases['d3conv221'],weights['d333conv1'],biases['d333conv1'])
        d3conv23=subblock(d3conv22,weights['d3conv231'],biases['d3conv231'],weights['d333conv1'],biases['d333conv1'])
        d3conv24=subblock(d3conv23,weights['d3conv241'],biases['d3conv241'],weights['d333conv1'],biases['d333conv1'])
        d3conv25=subblock(d3conv24,weights['d3conv251'],biases['d3conv251'],weights['d333conv2'],biases['d333conv2'])
        d3conv26=subblock(d3conv25,weights['d3conv261'],biases['d3conv261'],weights['d333conv2'],biases['d333conv2'])
        d3conv27=subblock(d3conv26,weights['d3conv271'],biases['d3conv271'],weights['d333conv2'],biases['d333conv2'])
        d3conv28=subblock(d3conv27,weights['d3conv281'],biases['d3conv281'],weights['d333conv2'],biases['d333conv2'])
        d3conv29=subblock(d3conv28,weights['d3conv291'],biases['d3conv291'],weights['d333conv2'],biases['d333conv2'])
        d3conv30=subblock(d3conv29,weights['d3conv301'],biases['d3conv301'],weights['d333conv2'],biases['d333conv2'])
        d3conv31=subblock(d3conv30,weights['d3conv311'],biases['d3conv311'],weights['d333conv2'],biases['d333conv2'])
        d3conv32=subblock(d3conv31,weights['d3conv321'],biases['d3conv321'],weights['d333conv2'],biases['d333conv2'])
        d3conv33=subblock(d3conv32,weights['d3conv331'],biases['d3conv331'],weights['d333conv2'],biases['d333conv2'])
        d3conv34=subblock(d3conv33,weights['d3conv341'],biases['d3conv341'],weights['d333conv2'],biases['d333conv2'])
        d3conv35=subblock(d3conv34,weights['d3conv351'],biases['d3conv351'],weights['d333conv2'],biases['d333conv2'])
        d3conv36=subblock(d3conv35,weights['d3conv361'],biases['d3conv361'],weights['d333conv2'],biases['d333conv2'])
        d3conv37=subblock(d3conv36,weights['d3conv371'],biases['d3conv371'],weights['d333conv2'],biases['d333conv2'])
        d3conv38=subblock(d3conv37,weights['d3conv381'],biases['d3conv381'],weights['d333conv2'],biases['d333conv2'])
        d3conv39=subblock(d3conv38,weights['d3conv391'],biases['d3conv391'],weights['d333conv2'],biases['d333conv2'])
        d3conv40=subblock(d3conv39,weights['d3conv401'],biases['d3conv401'],weights['d333conv2'],biases['d333conv2'])
        d3conv41=subblock(d3conv40,weights['d3conv411'],biases['d3conv411'],weights['d333conv2'],biases['d333conv2'])
        d3conv42=subblock(d3conv41,weights['d3conv421'],biases['d3conv421'],weights['d333conv2'],biases['d333conv2'])
        d3conv43=subblock(d3conv42,weights['d3conv431'],biases['d3conv431'],weights['d333conv2'],biases['d333conv2'])
        d3conv44=subblock(d3conv43,weights['d3conv441'],biases['d3conv441'],weights['d333conv2'],biases['d333conv2'])
        d3conv45=subblock(d3conv44,weights['d3conv451'],biases['d3conv451'],weights['d333conv2'],biases['d333conv2'])
        d3conv46=subblock(d3conv45,weights['d3conv461'],biases['d3conv461'],weights['d333conv2'],biases['d333conv2'])
        d3conv47=subblock(d3conv46,weights['d3conv471'],biases['d3conv471'],weights['d333conv2'],biases['d333conv2'])
        d3conv48=subblock(d3conv47,weights['d3conv481'],biases['d3conv481'],weights['d333conv2'],biases['d333conv2'])

    with tf.name_scope('Transition3'):
        t3=transition(d3conv48,weights['t3conv'],biases['t3conv'])

    with tf.name_scope('DenseBlock4'):
        d4conv1=subblock(t3,weights['d4conv11'],biases['d4conv11'],weights['d433conv1'],biases['d433conv1'])
        d4conv2=subblock(d4conv1,weights['d4conv21'],biases['d4conv21'],weights['d433conv1'],biases['d433conv1'])
        d4conv3=subblock(d4conv2,weights['d4conv31'],biases['d4conv31'],weights['d433conv1'],biases['d433conv1'])
        d4conv4=subblock(d4conv3,weights['d4conv41'],biases['d4conv41'],weights['d433conv1'],biases['d433conv1'])
        d4conv5=subblock(d4conv4,weights['d4conv51'],biases['d4conv51'],weights['d433conv1'],biases['d433conv1'])
        d4conv6=subblock(d4conv5,weights['d4conv61'],biases['d4conv61'],weights['d433conv1'],biases['d433conv1'])
        d4conv7=subblock(d4conv6,weights['d4conv71'],biases['d4conv71'],weights['d433conv1'],biases['d433conv1'])
        d4conv8=subblock(d4conv7,weights['d4conv81'],biases['d4conv81'],weights['d433conv1'],biases['d433conv1'])
        d4conv9=subblock(d4conv8,weights['d4conv91'],biases['d4conv91'],weights['d433conv1'],biases['d433conv1'])
        d4conv10=subblock(d4conv9,weights['d4conv101'],biases['d4conv101'],weights['d433conv1'],biases['d433conv1'])
        d4conv11=subblock(d4conv10,weights['d4conv111'],biases['d4conv111'],weights['d433conv1'],biases['d433conv1'])
        d4conv12=subblock(d4conv11,weights['d4conv121'],biases['d4conv121'],weights['d433conv1'],biases['d433conv1'])
        d4conv13=subblock(d4conv12,weights['d4conv131'],biases['d4conv131'],weights['d433conv1'],biases['d433conv1'])
        d4conv14=subblock(d4conv13,weights['d4conv141'],biases['d4conv141'],weights['d433conv1'],biases['d433conv1'])
        d4conv15=subblock(d4conv14,weights['d4conv151'],biases['d4conv151'],weights['d433conv1'],biases['d433conv1'])
        d4conv16=subblock(d4conv15,weights['d4conv161'],biases['d4conv161'],weights['d433conv1'],biases['d433conv1'])
        d4conv17=subblock(d4conv16,weights['d4conv171'],biases['d4conv171'],weights['d433conv2'],biases['d433conv2'])
        d4conv18=subblock(d4conv17,weights['d4conv181'],biases['d4conv181'],weights['d433conv2'],biases['d433conv2'])
        d4conv19=subblock(d4conv18,weights['d4conv191'],biases['d4conv191'],weights['d433conv2'],biases['d433conv2'])
        d4conv20=subblock(d4conv19,weights['d4conv201'],biases['d4conv201'],weights['d433conv2'],biases['d433conv2'])
        d4conv21=subblock(d4conv20,weights['d4conv211'],biases['d4conv211'],weights['d433conv2'],biases['d433conv2'])
        d4conv22=subblock(d4conv21,weights['d4conv221'],biases['d4conv221'],weights['d433conv2'],biases['d433conv2'])
        d4conv23=subblock(d4conv22,weights['d4conv231'],biases['d4conv231'],weights['d433conv2'],biases['d433conv2'])
        d4conv24=subblock(d4conv23,weights['d4conv241'],biases['d4conv241'],weights['d433conv2'],biases['d433conv2'])
        d4conv25=subblock(d4conv24,weights['d4conv251'],biases['d4conv251'],weights['d433conv2'],biases['d433conv2'])
        d4conv26=subblock(d4conv25,weights['d4conv261'],biases['d4conv261'],weights['d433conv2'],biases['d433conv2'])
        d4conv27=subblock(d4conv26,weights['d4conv271'],biases['d4conv271'],weights['d433conv2'],biases['d433conv2'])
        d4conv28=subblock(d4conv27,weights['d4conv281'],biases['d4conv281'],weights['d433conv2'],biases['d433conv2'])
        d4conv29=subblock(d4conv28,weights['d4conv291'],biases['d4conv291'],weights['d433conv2'],biases['d433conv2'])
        d4conv30=subblock(d4conv29,weights['d4conv301'],biases['d4conv301'],weights['d433conv2'],biases['d433conv2'])
        d4conv31=subblock(d4conv30,weights['d4conv311'],biases['d4conv311'],weights['d433conv2'],biases['d433conv2'])
        d4conv32=subblock(d4conv31,weights['d4conv321'],biases['d4conv321'],weights['d433conv2'],biases['d433conv2'])

    with tf.name_scope('ClassificationLayer'):
        out=classification(d4conv32,weights['fc'],biases['fc'])

    return out

predict_y=network(x)
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predict_y))
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimize=tf.train.AdamOptimizer().minimize(loss)

def save(path,sess):
    saver=tf.train.Saver()
    saver.save(sess,path)

def restore(sess):
    saver=tf.train.Saver()
    saver.restore(sess,tf.train.latest_checkpoint('./Dense-201-16-Shared/'))


def train():
    print('Loading train images')
    path='train/'
    files=os.listdir(path)
    images=[]
    for file in files: images.append(misc.imread(path+file))
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
    print('Batch size : 10')
    print('Variables saved after every 2 Epochs')
    print('100 Epochs')
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
            print('Epoch',epoch,'completed, loss :',epochloss)
            if epoch%2==0: save('Dense-201-16-Shared/var.ckpt',sess)
        print('Network trained')

train()
