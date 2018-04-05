# from __future__ import print_function
import tensorflow as tf
import imageio
import os
import numpy as np
import random
from PIL import Image
import glob
import random
import scipy

def readimage(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def readmask(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def readnormal(folder, index):
    path = os.path.join(folder, str(index)+'.png')
    image = imageio.imread(path)
    return image

def weight_variable(shape):
    """
    init the weight variables with truncated normal distribution
    """
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """
    init the bias term with constant
    """
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,w):
    """
    perfrom 2-d convolution
    
    inputs: x (h x w x in_depth)
    weights: w (kernel_size x kernel_size x in_depth x num_filters)
    
    return: output(h x w x num_filters)
    """
    return tf.nn.conv2d(x,w,strides=[1,1,1,1], padding = 'SAME')

def hourglass(inputs, in_dim, out_dim, inter_dim, conv1_size, conv2_size, conv3_size, conv4_size):
    """
    a particular layer of the hourglass architecture mensioned in the papaer
    inputs: example h x w x 128
    in_dim: input dimsion
    out_dim: output dimsion
    inter_dim: interim dimsion
    conv1_size: kernel size of the convolution layer 1, usually is square
    conv2_size: kernel size of the convolution layer 2
    conv3_size, conv4_size: the same as above
    The 1x1 conv before entering into the conv2, conv3, conv4 as mensioned in the papaer
    """
    w02 = weight_variable([1,1,in_dim,inter_dim])
    b02 = bias_variable([inter_dim])
    conv02 = tf.nn.relu(conv2d(inputs, w02) + b02)
    
    w03 = weight_variable([1,1,in_dim,inter_dim])
    b03 = bias_variable([inter_dim])
    conv03 = tf.nn.relu(conv2d(inputs, w03) + b03)
    
    w04 = weight_variable([1,1,in_dim,inter_dim])
    b04 = bias_variable([inter_dim])
    conv04 = tf.nn.relu(conv2d(inputs, w04) + b04)

    w1 = weight_variable([conv1_size,conv1_size,in_dim,out_dim//4])
    b1 = bias_variable([out_dim//4])
    conv1 = tf.nn.relu(conv2d(inputs,w1) + b1)   
    
    w2 = weight_variable([conv2_size,conv2_size,inter_dim,out_dim//4])
    b2 = bias_variable([out_dim//4])
    conv2 = tf.nn.relu(conv2d(conv02,w2) + b2)
    
    w3 = weight_variable([conv3_size,conv3_size,inter_dim, out_dim//4])
    b3 = bias_variable([out_dim//4])
    conv3 = tf.nn.relu(conv2d(conv03,w3) + b3)
    
    w4 = weight_variable([conv4_size,conv4_size,inter_dim, out_dim//4])
    b4 = bias_variable([out_dim//4])
    conv4 = tf.nn.relu(conv2d(conv04,w4) + b4)
    
    res = tf.concat([conv1, conv2, conv3, conv4], axis=3)
    # print(res.shape)
    
    return res


def buildModel(x):
    """
    build the stacked hourglass net model
    x: inputs 128 x 128 x 3
    return: outputs of the model 128 x 128 x 3
    """
    wh0 = weight_variable([3,3,3,128])
    bh0 = bias_variable([128])
    convH = tf.nn.relu(conv2d(x, wh0) + bh0)
    # print(convH.shape) # 128 x 128 x 3 -> 128 x 128 x 128
    #
    convA = hourglass(convH,128,64,64,1,3,7,11)
    # print(convA.shape) # 128 x 128 x 64
    [dummybatch,height4,width4,depth4] = convA.shape
    #
    #
    convB_maxpool = tf.nn.max_pool(convH, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')
    convB_1 = hourglass(convB_maxpool,128,128,32,1,3,5,7)
    convB_2 = hourglass(convB_1,128,128,32,1,3,5,7)
    # print(convB_2.shape) # 64 x 64 x 128

    ##
    convB_3 = hourglass(convB_2,128,128,32,1,3,5,7)
    convC = hourglass(convB_3,128,128,64,1,3,7,11)
    # print(convC.shape) # 64 x 64 x 128
    [dummybatch,height3,width3,depth3] = convC.shape
    ##
    ##
    convB_maxpool_2 = tf.nn.max_pool(convB_2, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')
    convB_4 = hourglass(convB_maxpool_2,128,128,32,1,3,5,7)
    convD = hourglass(convB_4,128,256,32,1,3,5,7)
    # print(convD.shape) # 32 x 32 x 256
    ##
    ###
    convE = hourglass(convD,256,256,32,1,3,5,7) 
    convF = hourglass(convE,256,256,64,1,3,7,11)
    # print(convF.shape) # 32 x 32 x 256
    [dummybatch,height2,width2,depth2] = convF.shape
    ###
    ###
    convD_maxpool = tf.nn.max_pool(convD, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')
    convE_2 = hourglass(convD_maxpool,256,256,32,1,3,5,7)
    convE_3 = hourglass(convE_2,256,256,32,1,3,5,7)
    # print(convE_3.shape) # 16 x 16 x 256
    ###
    ####
    convE_4 = hourglass(convE_3,256,256,32,1,3,5,7)
    convE_5 = hourglass(convE_4,256,256,32,1,3,5,7)
    # print(convE_5.shape) # 16 x 16 x 256
    [dummybatch,height,width,depth] = convE_5.shape
    ####
    ####
    convE_3_maxpool = tf.nn.max_pool(convE_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
    convE_6 = hourglass(convE_3_maxpool,256,256,32,1,3,5,7)
    convE_7 = hourglass(convE_6,256,256,32,1,3,5,7)
    convE_8 = hourglass(convE_7,256,256,32,1,3,5,7)
    # print(convE_8.shape) # 8 x 8 x 256
    ####
    ####
    upsample_4 = tf.image.resize_nearest_neighbor(convE_8,[height,width])
    convE_9 = tf.add(upsample_4,convE_5)
    # print(convE_9.shape) # 16 x 16 x 256
    ####
    ###
    convE_10 = hourglass(convE_9,256,256,32,1,3,5,7)
    convF_2 = hourglass(convE_10,256,256,64,1,3,7,11)
    upsample_3 = tf.image.resize_nearest_neighbor(convF_2,[height2,width2])
    convF_3 = tf.add(upsample_3,convF)
    #print(convF_3.shape)
    ###
    ##
    convE_11 = hourglass(convF_3,256,256,32,1,3,5,7)
    convG = hourglass(convE_11,256,128,32,1,3,5,7)
    upsample_2 = tf.image.resize_nearest_neighbor(convG,[height3,width3])
    convG_2 = tf.add(upsample_2,convC)
    #print(convG_2.shape)
    convB_5 = hourglass(convG_2,128,128,32,1,3,5,7)
    convA_2 = hourglass(convB_5,128,64,64,1,3,7,11)
    ##
    #
    upsample_1 = tf.image.resize_nearest_neighbor(convA_2,[height4,width4])
    convA_3 = tf.add(upsample_1,convA)
    #print(convA_3.shape)
    #
    wh1 = weight_variable([3,3,64,3])
    bh1 = bias_variable([3])
    convH_2 = tf.nn.relu(conv2d(convA_3, wh1) + bh1)
    return convH_2

def train_test_split(random_indexes,validation_size):
    train_indexes = random_indexes[:len(random_indexes)-validation_size]
    test_indexes = random_indexes[len(random_indexes)-validation_size:]
    return train_indexes, test_indexes

def get_batches(random_indexes, batch_size):
    num_batches = len(random_indexes) // batch_size
    indexes = random_indexes[:num_batches*batch_size]
    for idx in range(0, len(indexes),batch_size):
        yield indexes[idx:idx+batch_size]

def scan_png_files(folder):
    '''
    folder: 1.png 3.png 4.png 6.png 7.exr unknown.mpeg
    return: ['1.png', '3.png', '4.png']
    '''
    ext = '.png'
    ret = [fname for fname in os.listdir(folder) if fname.endswith(ext)]

    return ret


def evaluate(prediction_folder, groundtruth_folder, mask_folder):
    '''
    Evaluate mean angle error of predictions in the prediction folder,
    given the groundtruth and mask images.
    '''
    # Scan folders to obtain png files
    if mask_folder is None:
        mask_folder = os.path.join(groundtruth_folder, '..', 'mask')

    pred_pngs = scan_png_files(prediction_folder)
    gt_pngs = scan_png_files(groundtruth_folder)
    mask_pngs = scan_png_files(mask_folder)

    pred_diff_gt = set(pred_pngs).difference(gt_pngs)
    assert len(pred_diff_gt) == 0, \
        'No corresponding groundtruth file for the following files:\n' + '\n'.join(pred_diff_gt)
    pred_diff_mask = set(pred_pngs).difference(mask_pngs)
    assert len(pred_diff_mask) == 0, \
        'No corresponding mask file for the following files:\n' + '\n'.join(pred_diff_mask)

    # Measure: mean angle error over all pixels
    mean_angle_error = 0
    total_pixels = 0
    for fname in pred_pngs:
        # print('Proccessing file {}'.format(fname))
        prediction = imageio.imread(os.path.join(prediction_folder, fname))
        groundtruth = imageio.imread(os.path.join(groundtruth_folder, fname))
        mask = imageio.imread(os.path.join(mask_folder, fname)) # Greyscale image
       
        prediction = ((prediction*1.0 / 255.0) - 0.5) * 2
        groundtruth = ((groundtruth*1.0 / 255.0) - 0.5) * 2

        total_pixels += np.count_nonzero(mask)
        mask = mask != 0

        a11 = np.sum(prediction * prediction, axis=2)[mask]
        a22 = np.sum(groundtruth * groundtruth, axis=2)[mask]
        a12 = np.sum(prediction * groundtruth, axis=2)[mask]

        cos_dist = a12 / np.sqrt(a11 * a22)
        cos_dist[np.isnan(cos_dist)] = -1
        cos_dist = np.clip(cos_dist, -1, 1)

        angle_error = np.arccos(cos_dist)
        mean_angle_error += np.sum(angle_error)

    return mean_angle_error / total_pixels

data_size = 20000
epochs = 4
data = [i for i in range(data_size)]
batch_size = 20

train_color = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
train_mask = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
train_normal = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')

validation_color = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
validation_mask = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')
validation_normal = np.zeros(shape = (batch_size,128,128,3), dtype = 'float32')

# build the graph
train_graph = tf.Graph()
with train_graph.as_default():
    x = tf.placeholder('float32',[None, 128,128,3]) # color
    y = tf.placeholder('float32',[None, 128,128,3]) # mask
    z = tf.placeholder('float32',[None, 128,128,3]) # normal labels

    output = buildModel(x)

    mean_angle_error = 0
    total_pixels = 0

    for j in range(batch_size):        
        prediction = ((output[j,:,:,:]*1.0/255.0) - 0.5) * 2
        norm = ((z[j,:,:,:]*1.0/255.0) - 0.5) * 2
        mask = y[j,:,:,0]
        bmask = tf.cast(mask,tf.bool)

        total_pixels += tf.count_nonzero(bmask)
        
        a11 = tf.boolean_mask(tf.reduce_sum(prediction*prediction, axis=2),bmask)
        a22 = tf.boolean_mask(tf.reduce_sum(norm*norm, axis=2),bmask)
        a12 = tf.boolean_mask(tf.reduce_sum(prediction*norm, axis=2),bmask)

        cos_dist = -1.0*a12 / tf.sqrt(a11 * a22)
        # tf.assign(cos_dist[tf.is_nan(cos_dist)],-1) # missing this in the evalution
        cos_dist = tf.clip_by_value(cos_dist, -1, 1)
        # angle_error = tf.acos(cos_dist)
        mean_angle_error += tf.reduce_sum(cos_dist) # -1 the best

    cost = mean_angle_error*1.0 / tf.cast(total_pixels,tf.float32)
    opt = tf.train.AdamOptimizer(0.0001).minimize(cost)


# the driver
random.shuffle(data)
train, test = train_test_split(data,data_size//20)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    min_loss_so_far = 1.57
    for e in range(1,epochs+1):
        num_batches = 0
        los = 0
        every = 5
        for batch_index in get_batches(train,batch_size):
            counter = 0
            for i in batch_index:
                train_color[counter,:,:,:] = readimage('./train/color', i)
                train_mask[counter,:,:,0] = readmask('./train/mask', i)
                train_mask[counter,:,:,1] = readmask('./train/mask', i)
                train_mask[counter,:,:,2] = readmask('./train/mask', i)
                train_normal[counter,:,:,:] = readimage('./train/normal', i)
                counter += 1
            
            c, _ = sess.run([cost, opt], feed_dict={x: train_color, y:train_mask, z: train_normal})
            los += c
            num_batches += 1
            if num_batches % every == 0:
                print('Epoch {}/{};'.format(e,epochs),'Batches {}/{};'.format(num_batches,len(train)//batch_size),\
                      'Avg {} bathc(es) training loss: {:.3f}|{:.3f}'.format(every,los/every,np.arccos(-los/every)))
                los = 0

            if num_batches % 200 == 0:
                vlos = 0
                valid_batches = len(test) // batch_size
                for index in get_batches(test,batch_size):
                    counter = 0
                    for k in index:
                        validation_color[counter,:,:,:] = readimage('./train/color', k)
                        validation_mask[counter,:,:,0] = readmask('./train/mask', k)
                        validation_mask[counter,:,:,1] = readmask('./train/mask', k)
                        validation_mask[counter,:,:,2] = readmask('./train/mask', k)
                        validation_normal[counter,:,:,:] = readimage('./train/normal', k)
                        counter += 1
                    
                    vc = sess.run(cost, feed_dict={x: validation_color, y:validation_mask, z: validation_normal})
                    vlos += vc
                print('Avg validation loss: {:.3f}|{:.3f}\n'.format(vlos/valid_batches,np.arccos(-vlos/valid_batches)))
                if vlos < min_loss_so_far:
                    min_loss_so_far = vlos
                    tf.train.Saver().save(session, 'best_model')
                    print('best model saved!')

        # if e % 1 == 0:
        #     print('generate validation the picture')
        #     valid_color = np.zeros(shape = (1,128,128,3), dtype = 'float32')
        #     valid_mask = np.zeros(shape = (1,128,128,3), dtype = 'float32')
        #     # valid_normal = np.zeros(shape = (1,128,128,3), dtype = 'float32')
        #     cnt = 1
        #     for k in test:
        #         valid_color[0:,:,:] = readimage('./train/color', k)
        #         valid_mask[0,:,:,0] = readmask('./train/mask', k)
        #         valid_mask[0,:,:,1] = readmask('./train/mask', k)
        #         valid_mask[0,:,:,2] = readmask('./train/mask', k)
        #         result = sess.run(output, feed_dict = {x: valid_color, y:valid_mask})
        #         print(result.shape)
        #         # rescaled = 255.0 / np.max(result) * (result-np.min(result)).astype(np.uint8)
        #         image=Image.fromarray(result.astype(np.uint8)[0])
        #         # print(rescaled.shape)
        #         # image = Image.fromarray(rescaled[0])
        #         image.save('./train/pred/'+str(k)+'.png','png')
        #         cnt += 1
        #         if cnt % 100 == 0:
        #             print('generate',cnt,'pictures')
        #     valid = evaluate('./train/pred/', './train/normal/', './train/mask/')
        #     print(valid)


    # num_test = len(glob.glob('./test/color/*.png'))
    # test_color = np.zeros(shape = (1,128,128,3), dtype = 'float32')
    # test_mask = np.zeros(shape = (1,128,128,3), dtype = 'float32')
    # cnt = 1
    # for k in range(num_test):
    #     test_color[0,:,:,:] = readimage('./test/color', k)
    #     test_mask[0,:,:,0] = readmask('./test/mask', k)
    #     test_mask[0,:,:,1] = readmask('./test/mask', k)
    #     test_mask[0,:,:,2] = readmask('./test/mask', k)
    #     result = sess.run(output, feed_dict = {x:test_color,y:test_mask})
    #     # imgio = np.reshape(result,[128,128,3])
    #     # pth = './test/normal/'+str(k)+'.png'
    #     # scipy.misc.imsave(pth,imgio)
    #     image=Image.fromarray(result.astype(np.uint8)[0])
    #     image.save('./test/normal/'+str(k)+'.png','png')
    #     if cnt % 100 == 0:
    #         print('generate',cnt,'normal')
    #     cnt += 1
