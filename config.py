from pathlib import Path

def get_config():
    return {
        "number_x_values": 1000,                            #time series length
        "extra_tokens": ["ukn"],                            #additional tokens next to normal int index
        "vocab_size": 100000,                               #number of equally distanced pieces of number range [0,1], encoder interpolation CE 
        "remove_parts": True,                               #whether or not to remove arbitrary parts of the time series (corresponding to the entry "width_array")
        "spline_value": [800000,1100000],                                  #spline value for smooth time series shape, higher value -> more smooth
        "x_lim": [0,1000],                                  #range of x-array for removing parts of time series
        "y_lim": [10,10000],                                  #range of y-values in which time series is bounded
        # "noise_std": [0,10],                             #[noise_std_min, noise_std_max], noise_std is normal distributed or uniform distributed itself, specify in create_data.py in generate_noisy_data
        "noise_std": ["norm",0,0.15], #noise std is set to be normal or uniform distributed, alternative ["uni", 0.05 meaning 0.05 * max(TimeSeries)-min(TimeSeries), 0.15 meaning 0.15 * max(TimeSeries)-min(TimeSeries)], ["norm", 0 (mean), 0.15 (std, meaning 0.15 * max(TimeSeries)-min(TimeSeries))] 
        "dropout": 0.1,                                     #value for dropout layer in model
        "train_count":10000,                                    #number of training examples per epoch
        "val_count": 100,                                     #number of validation examples per epoch
        "random_number_range": ["norm",0,5],          #y-values are created randomly with normal distribution and given mean_value and std_value, alternative: ["uni", 5(lower border), 10(upper border)]
        "offset": 10,                                       #offset to the lowest and highest border of y_values
        "width_array_encoder": [10,100,10],                  #[min_width, max_width, max_count_width]
        "batch_size": 20,                                    #train_count / batch_size = number_iteration_per_epoch
        "num_epochs": 400,                                  #max number of training epochs
        "lr": 1e-4,                                       #learning rate for Adam optimizer
        "d_model": 512,
        "model_folder": "weights",                          #folder name to store models in
        "model_basename": "Encoder_Interpolation_Discontinuous_",
        "preload": "latest",                                #whether or not to start training with the latest trained version
        "experiment_name": "runs/tmodel",
        "label_smoothing": 0.0,
        "gradient_loss_weight": 0.2
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])