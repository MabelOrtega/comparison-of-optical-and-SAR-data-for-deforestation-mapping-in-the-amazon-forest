from utils import *
from networks import *
root_path = os.getcwd()

# folder to load config file
CONFIG_PATH = root_path
config = load_config(CONFIG_PATH, 'main_config.yaml')

exp = config['num_exp']
path_exp = root_path+'/' + config['folder_exp'] + '/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'

if not os.path.exists(path_exp):
    os.makedirs(path_exp)
if not os.path.exists(path_models):
    os.makedirs(path_models)
if not os.path.exists(path_maps):
    os.makedirs(path_maps)

loader = ImageLoader(config['image_type'])
image_stack = loader.load_images()
ref_2019 = load_tif_image(config['data_directory'] + '/r10m_def_2019.tif').astype('unit8')
past_ref = load_tif_image(config['data_directory'] + '/r10m_def_before_2019.tif').astype('unit8')
final_mask1 = mask_no_considered(ref_2019, config['buffer'], past_ref)
mask_tr_val = np.load(config['data_directory'] + '/mask_training_val.npy').astype('unit8')
mask_amazon_ts = np.load(config['data_directory'] + '/mask_testing.npy').astype('unit8')

h_, w_, channels = image_stack.shape
print('image stack size: ', image_stack.shape)

# Normalization
image_stack = normalization(image_stack.copy(), config['type_norm'])

# Print pertengate of each class (whole image)
print('Total no-deforestaion class is {}'.format(len(final_mask1[final_mask1==0])))
print('Total deforestaion class is {}'.format(len(final_mask1[final_mask1==1])))
print('Total past deforestaion class is {}'.format(len(final_mask1[final_mask1==1])))
print('Percentage of deforestaion class is {:.2f}'.format((len(final_mask1[final_mask1==1])*100)/len(final_mask1[final_mask1==0])))

# Creation of tile mask
mask_tiles = create_mask(final_mask1.shape[0], final_mask1.shape[1], grid_size=(5, 4))
image_stack = image_stack[:mask_tiles.shape[0], :mask_tiles.shape[1],:]
final_mask1 = final_mask1[:mask_tiles.shape[0], :mask_tiles.shape[1]]

# Create ixd image to extract patches
im_idx = create_idx_image(final_mask1)
patches_idx = extract_patches(im_idx, patch_size=(config['patch_size'], config['patch_size']), overlap=config['overlap']).reshape(-1, config['patch_size'], config['patch_size'])
patches_mask = extract_patches(mask_tr_val, patch_size=(config['patch_size'], config['patch_size']), overlap=config['overlap']).reshape(-1,config['patch_size'],config['patch_size'])
del im_idx

# Selecting index trn val and test patches idx
idx_trn = np.squeeze(np.where(patches_mask.sum(axis=(1, 2))==config['patch_size']**2))
idx_val = np.squeeze(np.where(patches_mask.sum(axis=(1, 2))==2*config['patch_size']**2))

patches_idx_trn = patches_idx[idx_trn]
patches_idx_val = patches_idx[idx_val]

print('Number of training patches:  ', len(patches_idx_trn), 'Number of validation patches', len(patches_idx_val))

# Extract patches with at least 2% of deforestation class
X_train = retrieve_idx_percentage(final_mask1, patches_idx_trn, pertentage = config['def_perdentage'])
X_valid = retrieve_idx_percentage(final_mask1, patches_idx_val, pertentage = config['def_perdentage'])

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                   vertical_flip = True)
valid_datagen = ImageDataGenerator(horizontal_flip = True,
                                   vertical_flip = True)

y_train = np.zeros((len(X_train)))
y_valid = np.zeros((len(X_valid)))

train_gen = train_datagen.flow(np.expand_dims(X_train, axis = -1), y_train,
                              batch_size=config['batch_size'],
                              shuffle=True)

valid_gen = valid_datagen.flow(np.expand_dims(X_valid, axis = -1), y_valid,
                              batch_size=config['batch_size'],
                              shuffle=False)

train_gen_batch = batch_generator(train_gen, image_stack, final_mask1, config['patch_size'], config['number_class'])
valid_gen_batch = batch_generator(valid_gen, image_stack, final_mask1, config['patch_size'], config['number_class'])

metrics_all = []

## computation graph
K.clear_session()
# parameters

input_shape_enc = (config['patch_size'], config['patch_size'], channels // 2)
input_shape_dec = (config['patch_size'], config['patch_size'], 1)

img_size, img_size, cdims = input_shape_enc
lr = 1e-3

# network definition
encoder = build_base_encoder(input_shape_enc)
encoder.summary()
decoder = build_base_decoder(input_shape_dec)
decoder.summary()

img_a = K.placeholder(dtype=tf.float32, name="img_a", shape=(config['batch_size'], img_size, img_size, cdims))
img_b = K.placeholder(dtype=tf.float32, name="img_b", shape=(config['batch_size'], img_size, img_size, cdims))
y_true = K.placeholder(dtype=tf.float32, name="y_true", shape=(config['batch_size'], img_size, img_size, 1))

[feat1_a, feat2_a, feat3_a, feat4_a] = encoder(img_a)
[feat1_b, feat2_b, feat3_b, feat4_b] = encoder(img_b)

distance1 = euclidean_distance(feat4_a, feat4_b)

distance = decoder([distance1, feat1_a, feat2_a, feat3_a, feat1_b, feat2_b, feat3_b])

# Parameters of the model
weights = [0.2, 0.8, 0]
loss = contrastive_loss(y_true, distance, weights)

t_vars = tf.trainable_variables()
net_vars = [var for var in t_vars if '_net' in var.name]
optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=net_vars)

sess = K.get_session()

init_op = tf.variables_initializer(net_vars)

# **************
for tm in range(0, config['times']):
    print('time: ', tm)

    sess.run(tf.global_variables_initializer())
    # print(net_vars)

    num_of_trn_batches = len(X_train) // config['batch_size']
    num_of_val_batches = len(X_valid) // config['batch_size']
    epochs = 100
    best_val_loss = np.inf
    # train the model

    for epoch in range(epochs):
        print('epoch :', epoch)
        trn_loss, trn_acc, t_dist = [], [], []
        start_time = time.time()
        for idx in range(0, num_of_trn_batches):

            # selecting a batch of images
            batch_t0, batch_t1, batch_ref = next(train_gen_batch)
            if batch_t0.shape[0] != config['batch_size']:
                continue

            feed_dict = {img_a: batch_t0, img_b: batch_t1, y_true: batch_ref}

            sess.run([optim], feed_dict=feed_dict)

            with sess.as_default():
                t_loss = loss.eval(feed_dict)
                # print('loss', t_loss)
                t_distance = distance.eval(feed_dict)
                trn_loss.append(t_loss)
                t_acc_cpu = accuracy_cpu(batch_ref, t_distance)
                trn_acc.append(t_acc_cpu)
                t_dist.append(t_distance)

                # Evaluating model on validation,
        val_loss, val_acc, v_dist = [], [], []
        for _ in range(0, num_of_val_batches):
            batch_t0, batch_t1, batch_ref = next(valid_gen_batch)
            if batch_t0.shape[0] != config['batch_size']:
                continue
            feed_dict = {img_a: batch_t0, img_b: batch_t1, y_true: batch_ref}
            with sess.as_default():
                v_loss = loss.eval(feed_dict)
                v_distance = distance.eval(feed_dict)
                val_loss.append(v_loss)
                v_acc_cpu = accuracy_cpu(batch_ref, v_distance)
                val_acc.append(v_acc_cpu)
                v_dist.append(v_distance)

        if best_val_loss > np.mean(val_loss):
            patience = 10
            best_val_loss = np.mean(val_loss)
            print('Saving best model and checkpoints')
            save_model(encoder, path_models + '/' + 'net_enc_' + str(tm) + '.h5')
            save_model(decoder, path_models + '/' + 'net_dec_' + str(tm) + '.h5')
            # Save the variables to disk.
            print('Ok')
        else:
            patience -= 1
        if patience < 0:
            break

        elapsed_time = time.time() - start_time
        print('loss trn: ', np.mean(trn_loss), 'Acc trn: ', np.mean(trn_acc), 'loss val: ', np.mean(val_loss),
              'Acc val: ', np.mean(val_acc))
        [feat1_a, feat2_a, feat3_a, feat4_a] = encoder.predict(config['batch_size'])
        [feat1_b, feat2_b, feat3_b, feat4_b] = encoder.predict(config['batch_size'])
        dist = euclidean_distance_np(feat4_a, feat4_b)
        up_distance = decoder.predict([dist, feat1_a, feat2_a, feat3_a, feat1_b, feat2_b, feat3_b])

    metrics_all.append(elapsed_time)

np.save(path_exp + '/metrics_tr.npy', metrics_all)