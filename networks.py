from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def resnet_block(x, n_filter, ind):
    x_init = x

    ## Conv 1
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name='res1_net' + str(ind))(x)
    x = Dropout(0.5, name='drop_net' + str(ind))(x)
    ## Conv 2
    x = Conv2D(n_filter, (3, 3), activation='relu', padding="same", name='res2_net' + str(ind))(x)

    ## Shortcut
    s = Conv2D(n_filter, (1, 3), activation='relu', padding="same", name='res3_net' + str(ind))(x_init)

    ## Add
    x = Add()([x, s])
    return x


def build_base_encoder(input_shape):
    '''Base network of the siamese encoder'''
    nb_filters = [32, 64, 128, 256]

    input_layer = Input(shape=input_shape, name="input_enc_net")

    res_block1 = resnet_block(input_layer, nb_filters[0], 1)
    pool1 = MaxPool2D((2, 2), name='pool_net1')(res_block1)

    res_block2 = resnet_block(pool1, nb_filters[1], 2)
    pool2 = MaxPool2D((2, 2), name='pool_net2')(res_block2)

    res_block3 = resnet_block(pool2, nb_filters[2], 3)
    pool3 = MaxPool2D((2, 2), name='pool_net3')(res_block3)

    res_block4 = resnet_block(pool3, nb_filters[2], 4)

    res_block5 = resnet_block(res_block4, nb_filters[2], 5)

    res_block6 = resnet_block(res_block5, nb_filters[2], 6)

    return Model(inputs=input_layer, outputs=[res_block1, res_block2, res_block3, res_block6])


def build_base_decoder(input_shape):
    '''Base network of the siamese decoder'''
    nb_filters = [1, 32, 64, 128]
    input_distance = Input(shape=(input_shape[0] // 8, input_shape[1] // 8, 1), name="input_dist_net")
    input_feat3_a = Input(shape=(input_shape[0], input_shape[1], nb_filters[1]), name="input_fa3_net")
    input_feat2_a = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, nb_filters[2]), name="input_fa2_net")
    input_feat1_a = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, nb_filters[3]), name="input_fa1_net")

    input_feat3_b = Input(shape=(input_shape[0], input_shape[1], nb_filters[1]), name="input_fb3_net")
    input_feat2_b = Input(shape=(input_shape[0] // 2, input_shape[1] // 2, nb_filters[2]), name="input_fb2_net")
    input_feat1_b = Input(shape=(input_shape[0] // 4, input_shape[1] // 4, nb_filters[3]), name="input_fb1_net")

    upsample3 = Conv2D(nb_filters[3], (3, 3), activation='relu', padding='same',
                       name='upsampling_net3')(UpSampling2D(size=(2, 2))(input_distance))

    merge3 = concatenate([upsample3, input_feat1_a, input_feat1_b], name='merge_net3')

    upsample2 = Conv2D(nb_filters[2], (3, 3), activation='relu', padding='same',
                       name='upsampling_net2')(UpSampling2D(size=(2, 2))(merge3))

    merge2 = concatenate([upsample2, input_feat2_a, input_feat2_b], name='merge_net2')

    upsample1 = Conv2D(nb_filters[1], (3, 3), activation='relu', padding='same',
                       name='upsampling_net1')(UpSampling2D(size=(2, 2))(merge2))

    merge1 = concatenate([upsample1, input_feat3_a, input_feat3_b], name='merge_net1')

    out = Conv2D(nb_filters[0], (1, 1), activation='relu', padding='same', name='out_net1')(merge1)
    return Model(inputs=[input_distance, input_feat3_a, input_feat2_a, input_feat1_a,
                         input_feat3_b, input_feat2_b, input_feat1_b], outputs=out)


def build_unet(input_shape, nb_filters, n_classes):
    '''Base network of the Unet architecture '''
    input_layer = Input(shape=input_shape)

    x = input_layer
    pools = []

    # Contracting Path
    for i in range(len(nb_filters)):
        x = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same', name=f'conv{i + 1}')(x)
        pools.append(x)
        x = MaxPool2D((2, 2))(x)

    x = Conv2D(nb_filters[-1], (3, 3), activation='relu', padding='same', name='conv_last')(x)

    # Expanding Path
    for i in range(len(nb_filters) - 1, 0, -1):
        x = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same', name=f'upsampling{i}')(
            UpSampling2D(size=(2, 2))(x))
        x = concatenate([pools[i - 1], x], name=f'concatenate{i}')

    output = Conv2D(n_classes, (1, 1), activation='softmax')(x)

    return Model(input_layer, output)


def build_resunet(input_shape, nb_filters, n_classes):
    '''Base network of the ResUnet architecture '''
    input_layer = Input(shape=input_shape, name="input_enc_net")
    x = input_layer

    res_blocks = []
    pool_layers = []
    for i in range(len(nb_filters)):
        res_blocks.append(resnet_block(x, nb_filters[i], i + 1))
        if i < len(nb_filters) - 1:
            pool_layers.append(MaxPool2D((2, 2), name=f'pool_net{i + 1}')(res_blocks[i]))
        x = pool_layers[-1] if pool_layers else res_blocks[i]

    for i in range(len(nb_filters) - 1, 0, -1):
        x = Conv2D(nb_filters[i], (3, 3), activation='relu', padding='same', name=f'upsampling_net{i}')(
            UpSampling2D(size=(2, 2))(x))
        x = concatenate([res_blocks[i - 1], x], name=f'concatenate{i}')

    output = Conv2D(n_classes, (1, 1), activation='softmax', padding='same', name='output')(x)

    return Model(input_layer, output)