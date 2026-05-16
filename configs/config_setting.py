import os

class setting_config:
    # Dataset configuration
    h5_path = 'data/fno_fpp_dataset.h5'
    target_size = 512
    train_ratio = 0.8
    val_ratio = 0.1
    batch_size = 16
    num_workers = 4

    # UFNO model configuration
    in_channels = 1
    out_channels = 1
    modes1 = 32
    modes2 = 32
    width = 128
    num_fno_layers = 2
    num_ufno_layers = 2

    # Training configuration
    epochs = 1000
    learning_rate = 1e-3
    weight_decay = 1e-4

    # Loss function configuration
    beta_smoothl1 = 0.1
    lamda_tgv = 0.001
    tgv_alpha0 = 1.0
    tgv_alpha1 = 2.0
    tgv_inner_iters = 5
    tgv_lr = 0.001

    # Save path configuration
    save_dir = 'checkpoints'

    @classmethod
    def create_save_dir(cls):
        if not os.path.exists(cls.save_dir):
            os.makedirs(cls.save_dir)