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

# Define model
input_shape = (config['patch_size'], config['patch_size'], channels)
nb_filters = [32, 64, 128]

if config['network'] == 'unet':
   model = build_unet(input_shape, nb_filters, config['number_class'])

if config['network'] == 'resunet':
   model = build_resunet(input_shape, nb_filters, config['number_class'])

# Parameters of the model
weights = [0.2, 0.8, 0]
adam = Adam(lr=config['lr'], beta_1=config['beta_1'])
loss = weighted_categorical_crossentropy(weights)

# Trainign loop
time_tr = []
for tm in range(0, config['times']):
    print('time: ', tm)

    model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])
    model.summary()

    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(path_models + '/' + config['network'] + '_' + str(tm) + '.h5', monitor='val_loss', verbose=1,
                                 save_best_only=True, mode='min')
    lr_reduce = ReduceLROnPlateau(factor=0.9, min_delta=0.0001, patience=5, verbose=1)
    callbacks_list = [earlystop, checkpoint]
    # train the model
    start_training = time.time()
    history = model.fit_generator(train_gen_batch,
                                  steps_per_epoch=len(X_train) * 3 // train_gen.batch_size,
                                  validation_data=valid_gen_batch,
                                  validation_steps=len(X_valid) * 3 // valid_gen.batch_size,
                                  epochs=100,
                                  callbacks=callbacks_list)
    end_training = time.time() - start_training
    time_tr.append(end_training)
time_tr_array = np.asarray(time_tr)
# Save training time
np.save(path_exp + '/metrics_tr.npy', time_tr_array)
