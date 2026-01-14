import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from create_data import callFunction
from useful import round_numbers_individually, calc_exp, round_with_exp, remove_parts_of_graph_encoder
from random import choice

#Dataset for encoder Interpolation
class TimeSeriesDataset_Interpolation_roundedInput(Dataset):
    def __init__(self, timeseries_count: int, x_values, config, v2i_dict) -> None:
        super().__init__()
        self.timeseries_count = timeseries_count
        self.x_values = x_values
        self.config = config
        self.value_to_index = v2i_dict

    def __len__(self):
        return self.timeseries_count
    
    def __getitem__(self, index):
        #random start value y_start within boundaries config["y_lim"][0]+1 and config["y_lim"][1]-1
        y_start = np.random.uniform(self.config["y_lim"][0]+1,self.config["y_lim"][1]-1)
        
        #calculate randomInt to select different time series generating functions
        # timeSeries = [0,1,2,3,4,5,6,7]
        # randomInt = choice(timeSeries)
        randomInt = 5
        #0: low order
        #1: low order
        #2: low order
        #3: periodic
        #4: high order
        #5: high order (periodic sum)
        #6: discontinuous 
        #7: discontinuous 

        #calculate timeseries
        y_spline, y_noise_spline,min_value, max_value, noise_std = callFunction(x_values=self.x_values, y_start=y_start, random_number_range=self.config["random_number_range"], spline_value=self.config["spline_value"], vocab_size=self.config["vocab_size"], randomInt=randomInt, noise_std=self.config["noise_std"])

        #remove arbitrary parts of timeseries
        mask = remove_parts_of_graph_encoder(self.x_values,y_noise_spline, self.config["width_array_encoder"], self.config["offset"], self.config["x_lim"])
        # mask = ~mask
        #mask index 0 -> keep
        #mask index 1 -> remove
        mask_indices = np.where(mask == 1)[0]

        #parameter for min-max scaling
        div_term = (max_value-min_value)          

        #perform min max scaling
        encoder_input = torch.tensor((y_noise_spline-min_value)/div_term,dtype=torch.float32)[:]
        encoder_input_removed = torch.tensor((y_noise_spline-min_value)/div_term,dtype=torch.float32)[:]
        decoder_input = torch.tensor((y_spline-min_value)/div_term,dtype=torch.float32)[:]
        decoder_input_copy = torch.tensor((y_spline-min_value)/div_term,dtype=torch.float32)[:]

        #calculate discrete values
        encoder_input = round_numbers_individually(self.config["vocab_size"],encoder_input)[:]
        encoder_input_removed = round_numbers_individually(self.config["vocab_size"],encoder_input_removed)[:]
        decoder_input_copy_rounded = round_numbers_individually(self.config["vocab_size"], decoder_input_copy)[:]

        #map discrete values to indices and map masked values to index of "ukn"-token
        exp = calc_exp(smallest_number=(1/self.config["vocab_size"]))
        encoder_input = encoder_input.apply_(lambda x: self.value_to_index[f"{round_with_exp(x, exp)}"]).type(torch.long)
        encoder_input_removed[mask_indices] = self.value_to_index["ukn"]
        encoder_input_removed = encoder_input_removed.apply_(lambda x: x if x>self.config["vocab_size"] else self.value_to_index[f"{round_with_exp(x, exp)}"]).type(torch.long)
        decoder_input_copy_rounded = decoder_input_copy_rounded.apply_(lambda x: self.value_to_index[f"{round_with_exp(x, exp)}"]).type(torch.long)


        return {
            "div_term": div_term,
            "min_value": min_value,
            "noisy_TimeSeries" : encoder_input,
            "interpolation_noisy_TimeSeries": encoder_input_removed,
            "groundTruth": decoder_input,
            "noise_std": noise_std,
            "groundTruth_indices": decoder_input_copy_rounded
        }

# #vector: vector indicating which value does not exist (1) and which values exist (0)
# #setting all rows and columns to corresponding value of vector
# #in case no vector is given, it returns a matrix full of True
# def causal_mask_timeSeries(size, vector: torch.Tensor = None):
#     mask = torch.zeros(1,size,size)
#     for i in range(size):
#         if vector[i] == 1:
#             # mask[0,i,:] = vector[i]
#             mask[0,:,i] = vector[i]
#     return mask == 0







