from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, \
    Dropout  # core内部定义了一系列常用的网络层，包括全连接、激活层等
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def get_unet(n_ch, patch_height, patch_width):
    inputs = Input(shape=(n_ch, patch_height, patch_width))
    # data_format：字符串，“channels_first”或“channels_last”之一，代表图像的通道维的位置。
    # 以128x128的RGB图像为例，“channels_first”应将数据组织为（3,128,128），而“channels_last”应将数据组织为（128,128,3）。该参数的默认值是~/.keras/keras.json中设置的值，若从未设置过，则为“channels_last”。
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    #
    # 1×1的卷积的作用
    # 大概有两个方面的作用：1. 实现跨通道的交互和信息整合2. 进行卷积核通道数的降维和升维。
    conv6 = Conv2D(2, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv6 = core.Reshape((2, patch_height * patch_width))(
        conv6)  # 此时output的shape是(batchsize,2,patch_height*patch_width)
    conv6 = core.Permute((2, 1))(conv6)  # 此时output的shape是(Npatch,patch_height*patch_width,2)即输出维度是(Npatch,2304,2)
    ############
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

    '''
    模型Model的compile方法:
	compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics = None, target_tensors=None)
	本函数编译模型以供训练，参数有
	optimizer：         优化器，为预定义优化器名或优化器对.可以在调用model.compile()之前初始化一个优化器对象，然后传入该函数。
	loss：              损失函数，为预定义损失函数名或一个目标函数
	metrics：           列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
	sample_weight_mode：如果需要按时间步为样本赋权（2D权矩阵），将该值设为“temporal”。默认为“None”，代表按样本赋权（1D权）。如果模型有多个输出，可以向该参数传入指定sample_weight_mode的字典或列表。在下面fit函数的解释中有相关的参考内容。
	weighted_metrics:   metrics列表，在训练和测试过程中，这些metrics将由sample_weight或clss_weight计算并赋权
	target_tensors:     默认情况下，Keras将为模型的目标创建一个占位符，该占位符在训练过程中将被目标数据代替。如果你想使用自己的目标张量（相应的，Keras将不会在训练时期望为这些目标张量载入外部的numpy数据），你可以通过该参数手动指定。目标张量可以是一个单独的张量（对应于单输出模型），也可以是一个张量列表，或者一个name->tensor的张量字典。
	kwargs：            使用TensorFlow作为后端请忽略该参数，若使用Theano/CNTK作为后端，kwargs的值将会传递给 K.function。如果使用TensorFlow为后端，这里的值会被传给tf.Session.run
	在Keras中，compile主要完成损失函数和优化器的一些配置，是为训练服务的。
    '''
