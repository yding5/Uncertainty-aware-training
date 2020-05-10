
import os
import sys
sys.path.append("..")
from util.config import Config


def get_config_3d_heart17_8class_largerFit():
    # Set your own path, if needed.
    data_root_dir = os.path.abspath('data')  # The path where the downloaded dataset is stored.
    dataset_folder = "heart17_sameSize"   ## the name of the dataset folder
    dataset_dir = os.path.join(data_root_dir, dataset_folder)
    preprocess_folder = "preprocessed_8class_largerFit"  # The path where the downloaded dataset is stored.
    preprocess_dir = os.path.join(dataset_dir, preprocess_folder)
    data_folder = "data_origin"  # The path where the downloaded dataset is stored.
    data_dir = os.path.join(dataset_dir, data_folder)
    base_dir = os.path.abspath('output_experiment')

    c = Config(
        exp_ID = '0',
        loss_type = 1,
        loss_weight = 2000,
        save_image = True,

        update_from_argv=True,
        expName="8class_largerFit",

        ## network parameters
        num_classes=8,  ## the number of classes of input and output
        in_channels=1,   ## 3 for rgb or 1 for others
        segWeights=[1,1,1,1,1,2,1,1],
        batch_size=8,    ## batch size 
        patch_size=64,   ## the input size 64*64*64
        n_epochs=10,      ## the number of running epoches
        learning_rate=0.0002, ## learning rate
        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        initial_filter_size = 32, # the initial_filter_size of unet

        ## hardware info
        device="cuda",#"cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        ## Logging parameters
        name=dataset_folder+"_"+preprocess_folder,  ## subfix of output folder name
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=True,
        use_explogger=False,  

        ## Turn it to Trus, when you need to restore the model for further training or just for test
        checkpoint_file = "test",
        do_load_checkpoint=False,
        checkpoint_dir='data/checkpoint/checkpoint_last.pth.tar',

        data_root_dir = data_root_dir,
        dataset_dir = dataset_dir,
        data_dir = data_dir,
        preprocess_dir = preprocess_dir,
        data_train_dir = preprocess_dir, #os.path.join(dataset_dir, preprocess_folder),  # This is where your training data is stored
        data_test_dir = preprocess_dir, #os.path.join(dataset_dir, preprocess_folder), # This is where your validation data is stored
        data_vali_dir = preprocess_dir, #os.path.join(dataset_dir, preprocess_folder),  # This is where your test data is stored
        # output dir
        base_dir=base_dir,  # Where to log the output of the experiment.
        # cross validation configuration
        cross_vali_N = 5,
        cross_vali_index=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.
        cross_vali_dir= dataset_dir,  # This is where the 'splits.pkl' file is located, that holds your splits.
        cross_vali_result_all_dir = 'data/checkpoint/'+"crossvali_"+dataset_folder+"_"+preprocess_folder

    )
    print(c)
    return c
