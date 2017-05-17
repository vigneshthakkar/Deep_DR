import tensorflow as tf

initializer=tf.contrib.layers.xavier_initializer()
saver=tf.train.Saver()

weights={
    'conv1': tf.get_variable('conv1w',[3,3,1,16],tf.float32,initializer),

    'conv21': tf.get_variable('conv21w',[3,3,16,160],tf.float32,initializer),
    'conv22': tf.get_variable('conv22w',[3,3,160,160],tf.float32,initializer),
    'conv23': tf.get_variable('conv23w',[3,3,160,160],tf.float32,initializer),
    'conv24': tf.get_variable('conv24w',[3,3,160,160],tf.float32,initializer),

    'conv31': tf.get_variable('conv31w',[3,3,160,320],tf.float32,initializer),
    'conv32': tf.get_variable('conv32w',[3,3,320,320],tf.float32,initializer),
    'conv33': tf.get_variable('conv33w',[3,3,320,320],tf.float32,initializer),
    'conv34': tf.get_variable('conv34w',[3,3,320,320],tf.float32,initializer),

    'conv41': tf.get_variable('conv41w',[3,3,320,640],tf.float32,initializer),
    'conv42': tf.get_variable('conv42w',[3,3,640,640],tf.float32,initializer),
    'conv43': tf.get_variable('conv43w',[3,3,640,640],tf.float32,initializer),
    'conv44': tf.get_variable('conv44w',[3,3,640,640],tf.float32,initializer),
    '''
    'conv51': tf.get_variable('conv51w',[3,3,320,400],tf.float32,initializer),
    'conv52': tf.get_variable('conv52w',[3,3,400,400],tf.float32,initializer),
    'conv53': tf.get_variable('conv53w',[3,3,400,400],tf.float32,initializer),
    'conv54': tf.get_variable('conv54w',[3,3,400,400],tf.float32,initializer),

    'conv61': tf.get_variable('conv61w',[3,3,400,480],tf.float32,initializer),
    'conv62': tf.get_variable('conv62w',[3,3,480,480],tf.float32,initializer),
    'conv63': tf.get_variable('conv63w',[3,3,480,480],tf.float32,initializer),
    'conv64': tf.get_variable('conv64w',[3,3,480,480],tf.float32,initializer),

    'conv71': tf.get_variable('conv71w',[3,3,480,560],tf.float32,initializer),
    'conv72': tf.get_variable('conv72w',[3,3,560,560],tf.float32,initializer),
    'conv73': tf.get_variable('conv73w',[3,3,560,560],tf.float32,initializer),
    'conv74': tf.get_variable('conv74w',[3,3,560,560],tf.float32,initializer),

    'conv81': tf.get_variable('conv81w',[3,3,560,640],tf.float32,initializer),
    'conv82': tf.get_variable('conv82w',[3,3,640,640],tf.float32,initializer),
    'conv83': tf.get_variable('conv83w',[3,3,640,640],tf.float32,initializer),
    'conv84': tf.get_variable('conv84w',[3,3,640,640],tf.float32,initializer),
    '''
    'fc': tf.get_variable('fcw',[640,5],tf.float32,initializer)
}

biases={
    'conv1': tf.get_variable('conv1b',[16],tf.float32,initializer),

    'conv21': tf.get_variable('conv21b',[160],tf.float32,initializer),
    'conv22': tf.get_variable('conv22b',[160],tf.float32,initializer),
    'conv23': tf.get_variable('conv23b',[160],tf.float32,initializer),
    'conv24': tf.get_variable('conv24b',[160],tf.float32,initializer),

    'conv31': tf.get_variable('conv31b',[320],tf.float32,initializer),
    'conv32': tf.get_variable('conv32b',[320],tf.float32,initializer),
    'conv33': tf.get_variable('conv33b',[320],tf.float32,initializer),
    'conv34': tf.get_variable('conv34b',[320],tf.float32,initializer),

    'conv41': tf.get_variable('conv41b',[640],tf.float32,initializer),
    'conv42': tf.get_variable('conv42b',[640],tf.float32,initializer),
    'conv43': tf.get_variable('conv43b',[640],tf.float32,initializer),
    'conv44': tf.get_variable('conv44b',[640],tf.float32,initializer),
    '''
    'conv51': tf.get_variable('conv51b',[400],tf.float32,initializer),
    'conv52': tf.get_variable('conv52b',[400],tf.float32,initializer),
    'conv53': tf.get_variable('conv53b',[400],tf.float32,initializer),
    'conv54': tf.get_variable('conv54b',[400],tf.float32,initializer),

    'conv61': tf.get_variable('conv61b',[480],tf.float32,initializer),
    'conv62': tf.get_variable('conv62b',[480],tf.float32,initializer),
    'conv63': tf.get_variable('conv63b',[480],tf.float32,initializer),
    'conv64': tf.get_variable('conv64b',[480],tf.float32,initializer),

    'conv71': tf.get_variable('conv71b',[560],tf.float32,initializer),
    'conv72': tf.get_variable('conv72b',[560],tf.float32,initializer),
    'conv73': tf.get_variable('conv73b',[560],tf.float32,initializer),
    'conv74': tf.get_variable('conv74b',[560],tf.float32,initializer),

    'conv81': tf.get_variable('conv81b',[640],tf.float32,initializer),
    'conv82': tf.get_variable('conv82b',[640],tf.float32,initializer),
    'conv83': tf.get_variable('conv83b',[640],tf.float32,initializer),
    'conv84': tf.get_variable('conv84b',[640],tf.float32,initializer),
    '''
    'fc': tf.get_variable('fcb',[5],tf.float32,initializer)
}

x=tf.placeholder(tf.float32)
y=tf.placeholder(tf.float32)

def conv(input,weight,bias,istraining,name):
    with tf.name_scope(name):
        batchnorm=tf.contrib.layers.batch_norm(input,center=True,Scale=True,is_training=istraining)
        activate=tf.nn.relu(batchnorm)
        convolution=tf.add(tf.nn.conv2d(activate,weight,strides=[1,1,1,1],padding='SAME'),bias)
        return convolution

def network(x,istraining):

    with tf.name_scope('Block 1'):
        conv1=tf.add(tf.nn.conv2d(x,weights['conv1'],strides=[1,1,1,1], padding='SAME')+biases['conv1'],name='conv1')

    with tf.name_scope('Block 2'):
        conv21=conv(conv1,weights['conv21'],biases['conv21'],istraining,'conv21')
        conv22=tf.add(conv(conv21,weights['conv22'],biases['conv22'],istraining,'conv22')+conv1)
        conv23=conv(conv22,weights['conv23'],biases['conv23'],istraining,'conv23')
        conv24=tf.add(conv(conv23,weights['conv24'],biases['conv24'],istraining,'conv24')+conv22)
        avgpool2=tf.nn.avg_pool(conv24,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool2')

    with tf.name_scope('Block 3'):
        conv31=conv(avgpool2,weights['conv31'],biases['conv31'],istraining,'conv31')
        conv32=tf.add(conv(conv31,weights['conv32'],biases['conv32'],istraining,'conv32')+avgpool2)
        conv33=conv(conv32,weights['conv33'],biases['conv33'],istraining,'conv33')
        conv34=tf.add(conv(conv33,weights['conv34'],biases['conv34'],istraining,'conv34')+conv32)
        avgpool3=tf.nn.avg_pool(conv34,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool3')
    '''
    with tf.name_scope('Block 4'):
        conv41=conv(avgpool3,weights['conv41'],biases['conv41'],istraining,'conv41')
        conv42=tf.add(conv(conv41,weights['conv42'],biases['conv42'],istraining,'conv42')+avgpool3)
        conv43=conv(conv42,weights['conv43'],biases['conv43'],istraining,'conv43')
        conv44=tf.add(conv(conv43,weights['conv44'],biases['conv44'],istraining,'conv44')+conv42)
        avgpool4=tf.nn.avg_pool(conv44,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool4')

    with tf.name_scope('Block 5'):
        conv51=conv(avgpool4,weights['conv51'],biases['conv51'],istraining,'conv51')
        conv52=tf.add(conv(conv51,weights['conv52'],biases['conv52'],istraining,'conv52')+avgpool4)
        conv53=conv(conv52,weights['conv53'],biases['conv53'],istraining,'conv53')
        conv54=tf.add(conv(conv53,weights['conv54'],biases['conv54'],istraining,'conv54')+conv52)
        avgpool5=tf.nn.avg_pool(conv54,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool5')

    with tf.name_scope('Block 6'):
        conv61=conv(avgpool5,weights['conv61'],biases['conv61'],istraining,'conv61')
        conv62=tf.add(conv(conv61,weights['conv62'],biases['conv62'],istraining,'conv62')+avgpool5)
        conv63=conv(conv62,weights['conv63'],biases['conv63'],istraining,'conv63')
        conv64=tf.add(conv(conv63,weights['conv64'],biases['conv64'],istraining,'conv64')+conv62)
        avgpool6=tf.nn.avg_pool(conv64,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool6')

    with tf.name_scope('Block 7'):
        conv71=conv(avgpool6,weights['conv71'],biases['conv71'],istraining,'conv71')
        conv72=tf.add(conv(conv71,weights['conv72'],biases['conv72'],istraining,'conv72')+avgpool6)
        conv73=conv(conv72,weights['conv73'],biases['conv73'],istraining,'conv73')
        conv74=tf.add(conv(conv73,weights['conv74'],biases['conv74'],istraining,'conv74')+conv72)
        avgpool7=tf.nn.avg_pool(conv74,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool7')

    with tf.name_scope('Block 8'):
        conv81=conv(avgpool7,weights['conv81'],biases['conv81'],istraining,'conv81')
        conv82=tf.add(conv(conv81,weights['conv82'],biases['conv82'],istraining,'conv82')+avgpool7)
        conv83=conv(conv82,weights['conv83'],biases['conv83'],istraining,'conv83')
        conv84=tf.add(conv(conv83,weights['conv84'],biases['conv84'],istraining,'conv84')+conv82)
        avgpool8=tf.nn.avg_pool(conv84,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='avgpool8')
    '''
    with tf.name_scope('fc'):
        avgpool8=tf.reshape(avgpool8,[-1,1*1*640])
        fc=tf.add(tf.matmul(avgpool8,weights['fc']),biases['fc'])

    return fc

def predict():
    output=network(x,False)
    predict_y=tf.nn.softmax(output)
    return predict_y

def train():
    output=network(x,True)
    with tf.name_scope('Loss'):
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer=tf.train.AdamOptimizer().minimize(loss)
    return loss,optimizer

def save(path,sess):
    saver.save(sess,path)

def restore(path,sess):
    saver.restore(sess,path)
