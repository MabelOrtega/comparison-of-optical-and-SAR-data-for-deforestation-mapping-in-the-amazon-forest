from utils import *
from networks import *
root_path = os.getcwd()
print(root_path)

# folder to load config file
CONFIG_PATH = root_path
config = load_config(CONFIG_PATH, 'main_config.yaml')

exp = config['num_exp']
path_exp = root_path+'/' + config['folder_exp'] + '/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'

loader = ImageLoader(config['image_type'])
image_stack = loader.load_images()
image_stack = normalization(image_stack.copy(), config['type_norm'])
ref_2019 = load_tif_image(config['data_directory'] + '/r10m_def_2019.tif').astype('unit8')
past_ref = load_tif_image(config['data_directory'] + '/r10m_def_before_2019.tif').astype('unit8')
final_mask1 = mask_no_considered(ref_2019, config['buffer'], past_ref)
mask_tr_val = np.load(config['data_directory'] + '/mask_training_val.npy').astype('unit8')
mask_amazon_ts = np.load(config['data_directory'] + '/mask_testing.npy').astype('unit8')

time_ts = []
n_pool = 3
n_rows = 5
n_cols = 4
rows, cols = image_stack.shape[:2]
pad_rows = rows - np.ceil(rows / (n_rows * 2 ** n_pool)) * n_rows * 2 ** n_pool
pad_cols = cols - np.ceil(cols / (n_cols * 2 ** n_pool)) * n_cols * 2 ** n_pool
print(pad_rows, pad_cols)

npad = ((0, int(abs(pad_rows))), (0, int(abs(pad_cols))), (0, 0))
image1_pad = np.pad(image_stack, pad_width=npad, mode='reflect')

h, w, c = image1_pad.shape
patch_size_rows = h // n_rows
patch_size_cols = w // n_cols
num_patches_x = int(h / patch_size_rows)
num_patches_y = int(w / patch_size_cols)


nb_filters = [32, 64, 128]

if config['save_prob'] == False:
    prob_rec = np.zeros((image_stack.shape[0], image_stack.shape[1], config['times']), dtype=np.float32)

if config['network'] == 'unet':
    input_shape = (patch_size_rows, patch_size_cols, c)
    new_model = build_unet(input_shape, nb_filters, config['number_class'])

if config['network'] == 'resunet':
    input_shape = (patch_size_rows, patch_size_cols, c)
    new_model = build_resunet(input_shape, nb_filters, config['number_class'])

if config['network'] == 'siamese':
    input_shape = (patch_size_rows, patch_size_cols, c//2)
    new_enc = build_base_encoder(input_shape)
    new_dec = build_base_decoder((patch_size_rows, patch_size_cols, 1))

# Test loop
for tm in range(0, config['times']):
    print('time: ', tm)

    if config['network'] == 'unet' or 'resunet':
        model = load_model(path_models + '/' + config['network'] + '_' + str(tm) + '.h5', compile=False)

        for l in range(1, len(model.layers)):
            new_model.layers[l].set_weights(model.layers[l].get_weights())

        start_test = time.time()
        patch_t = []

        for i in range(0, num_patches_y):
            for j in range(0, num_patches_x):
                patch = image1_pad[patch_size_rows * j:patch_size_rows * (j + 1), patch_size_cols * i:patch_size_cols * (i + 1), :]
                predictions_ = new_model.predict(np.expand_dims(patch, axis=0))
                del patch
                patch_t.append(predictions_[:, :, :, 1])
                del predictions_
    else:
        model = load_model(path_models + '/' + 'net_enc_' + str(tm) + '.h5', compile=False)
        for l in range(1, len(model.layers)):
            new_enc.layers[l].set_weights(model.layers[l].get_weights())

        dec_model = load_model(path_models + '/' + 'net_dec_' + str(tm) + '.h5', compile=False)
        for l in range(1, len(dec_model.layers)):
            new_dec.layers[l].set_weights(dec_model.layers[l].get_weights())

        start_test = time.time()
        patch_t = []

        for i in range(0, num_patches_y):
            for j in range(0, num_patches_x):
                patch = image1_pad[patch_size_rows * j:patch_size_rows * (j + 1),patch_size_cols * i:patch_size_cols * (i + 1), :]
                [pred1_a, pred2_a, pred3_a, pred4_a] = new_enc.predict(np.expand_dims(patch[:, :, :c // 2], axis=0))
                [pred1_b, pred2_b, pred3_b, pred4_b] = new_enc.predict(np.expand_dims(patch[:, :, c // 2:], axis=0))
                dist = euclidean_distance_np(pred4_a, pred4_b)
                up_distance = new_dec.predict([dist, pred1_a, pred2_a, pred3_a, pred1_b, pred2_b, pred3_b])
                print(up_distance.shape)
                del patch
                patch_t.append(up_distance[:, :, :, 0])
                del dist

    end_test = time.time() - start_test
    patches_pred = np.asarray(patch_t).astype(np.float32)

    prob_recontructed = pred_reconctruct(h, w, num_patches_x, num_patches_y, patch_size_rows, patch_size_cols, patches_pred)
    prob_recontructed = prob_recontructed[:image_stack.shape[0], :image_stack.shape[1]]
    if config['save_prob']:
        np.save(path_maps + '/' + 'prob_map_' + str(tm) + '.npy', prob_recontructed)

    else:
        prob_rec[:, :, tm] = prob_recontructed

    time_ts.append(end_test)
    del prob_recontructed, model, patches_pred

time_ts_array = np.asarray(time_ts)
# Save test time
np.save(path_exp + '/metrics_ts.npy', time_ts_array)

if config['save_prob']:
    prob_rec = np.zeros((image_stack.shape[0], image_stack.shape[1], config['times']), dtype = np.float32)
    for tm in range (0, config['times']):
        prob_rec[:,:,tm] = np.load(path_maps + '/' + 'prob_map_' + str(tm) + '.npy').astype(np.float32)

# Saving average prediction map
mean_prob = np.mean(prob_rec, axis = -1)
np.save(path_maps+'/prob_mean.npy', mean_prob)

# Computing metrics
mean_prob = mean_prob[:final_mask1.shape[0], :final_mask1.shape[1]]
ref1 = np.ones_like(final_mask1).astype(np.float32)

ref1[final_mask1 == 2] = 0
TileMask = mask_amazon_ts * ref1
GTTruePositives = final_mask1 == 1

Npoints = 50
Pmax = np.max(mean_prob[GTTruePositives * TileMask == 1])
ProbList = np.linspace(Pmax, 0, Npoints)

metrics_ = matrics_AA_recall(ProbList, mean_prob, final_mask1, mask_amazon_ts, 625)
np.save(path_exp + '/acc_metrics.npy', metrics_)

# Comput Mean Average Precision (mAP) score
Recall = metrics_[:, 0]
Precision = metrics_[:, 1]
AA = metrics_[:, 2]

DeltaR = Recall[1:] - Recall[:-1]
AP = np.sum(Precision[:-1] * DeltaR)
print('mAP', AP)

