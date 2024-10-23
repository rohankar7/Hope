random_seed = 42
triplane_resolution = 256
triplane_features = 4
triplane_dir = f'./triplane_images_{triplane_resolution}_alpha'
pwd = 'C:/ShapeNetCore'
# ShapeNetCore_classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657', '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257', '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134', '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244', '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243', '04401088', '04460130', '04468005', '04530566', '04554684']
suffix_dir = 'models/model_normalized.obj'
voxel_resolution = 64
voxel_dir = f'./voxel_data_{voxel_resolution}'
voxel_type = 'grayscale' # or 'color'
captions = './text/ShapeNetCore_Captions.csv'

vae_weights_dir = './vae_weights'
latent_dir = f'./latent_images_{triplane_resolution}'
vae_resolution = 32
vae_features = 3
vae_lr = 3e-4
mlp_weights_dir = './mlp_weights'

# generated models
generation_dir = f'./generated_models_{triplane_resolution}'