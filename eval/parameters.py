# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(version):
    params = dict(
      
        dataset_dir = '/home/data/kbh/DCASE2022_SELD_dataset/',
        #dataset_dir = '/home/data/kbh/DCASE2022_SELD_synth_data/',

        # OUTPUT PATHS
        # feat_label_dir="DCASE2020_SELD_dataset/feat_label_hnet/",  # Directory to dump extracted features and labels
        feat_label_dir="/home/data/kbh/DCASE_output/"+version+"/feat",
        dcase_output_dir="/home/data/kbh/DCASE2020/out/"+version+"/",   
        # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',         # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',       # 'foa' - ambisonic or 'mic' - microphone signals

        #FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        # default : 60
        max_audio_len_s=60,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # METRIC
        average = 'macro',        # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

        ## ????
        label_sequence_length=50,    # Feature sequence length
        use_salsalite = False, # Used for MIC dataset only. If true use salsalite features, else use GCC features
        nb_mel_bins=64,



    )

    # Fixed n_class
    params['unique_classes'] = 13
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14 
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
