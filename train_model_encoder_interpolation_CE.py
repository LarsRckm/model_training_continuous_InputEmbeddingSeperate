import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
import warnings
import os

from model import build_encoder_interpolation_uknToken_projection
from config import *
from dataset_timeSeries import TimeSeriesDataset_Interpolation_roundedInput
from useful import value_to_index_dict, index_to_value_dict


def get_ds_timeSeries(config):
    train_count = config["train_count"]
    val_count = config["val_count"]
    x_values = np.arange(0, config["number_x_values"])
    v2i_dict = value_to_index_dict(config["vocab_size"], config["extra_tokens"])

    train_ds = TimeSeriesDataset_Interpolation_roundedInput(train_count, x_values, config, v2i_dict)
    val_ds = TimeSeriesDataset_Interpolation_roundedInput(val_count, x_values, config, v2i_dict)

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'])
    val_dataloader = DataLoader(val_ds, batch_size=1)

    return train_dataloader, val_dataloader, x_values.shape[0], len(v2i_dict)

def get_model_timeSeries(config, seq_len, vocab_size):
    model = build_encoder_interpolation_uknToken_projection(vocab_size,seq_len, config["d_model"], dropout=config["dropout"])
    return model

def run_validation_TimeSeries(model,validation_dl, device, num_examples, config, epoch_nr):
    model.eval()
    count = 0
    df = pd.DataFrame()

    with torch.no_grad():
        for batch in validation_dl:
            encoder_input = batch['noisy_TimeSeries'].to(device)                        #(Batch,seq_len) --> index shape
            encoder_input_removed = batch['interpolation_noisy_TimeSeries'].to(device)  #(Batch,seq_len) --> index shape
            noise_std = batch["noise_std"]                                              #(Batch) --> float shape
            div_term = batch["div_term"].to(device)                                     #(Batch) --> float shape
            min_value = batch["min_value"].to(device)
            time = torch.linspace(0, 1, steps=1000).unsqueeze(0).to(device)                                  #(Batch) --> float shape

            assert encoder_input.size(0) == 1, "Batch size needs to be 1"

            if(config["remove_parts"]):              
                model_out = greedy_decode_timeSeries_paper(model, encoder_input_removed, time)
            else:
                model_out = greedy_decode_timeSeries_paper(model, encoder_input, time)

            decoder_input = batch['groundTruth'].to(device)
            decoder_input = (div_term*decoder_input)+min_value

            df.loc[:,f"noise_{count}"] = encoder_input[0,:].cpu().numpy()                   #index form
            df.loc[:,f"noise_removed_{count}"] = encoder_input_removed[0,:].cpu().numpy()   #index form
            df.loc[:,f"groundTruth_{count}"] = decoder_input[0,:].cpu().numpy()             #float form
            df.loc[:,f"prediction_{count}"] = model_out[:].cpu().numpy()                    #index form
            df.loc[0,f"min_value_{count}"] = min_value[0].cpu().numpy()                     #float form
            df.loc[0,f"div_term_{count}"] = div_term[0].cpu().numpy()                       #float form
            df.loc[0,f"noise_std_{count}"] = noise_std[0].cpu().numpy()                     #float form
            count +=1
            if count == num_examples:
                df.to_csv(f"results_val/val_epoch_{epoch_nr}.csv", index=False)
                break

def greedy_decode_timeSeries_paper(model, source: torch.Tensor, time: torch.Tensor):
    
    encoder_output = model.encode(source, None, time)
    
    proj_out = model.project(encoder_output[0,:])           #(seq_len, d_model) -> (seq_len, vocab_size)

    _, indices = torch.max(proj_out, dim=1)                 #(batch, seq_len, vocab_size) -> (seq_len)
    
    return indices


def grad_norm(loss, model: nn.Module):
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=True,
        create_graph=False,
        allow_unused=True
    )
    total = 0.0
    for g in grads:
        if g is not None:
            total += g.detach().pow(2).sum()
    return total.sqrt()

def train_model_TimeSeries_paper(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    os.makedirs("weights/", exist_ok=True)
    os.makedirs("results_train", exist_ok=True)
    os.makedirs("results_val", exist_ok=True)
    
    train_dataloader, val_dataloader, seq_len, vocab_size = get_ds_timeSeries(config)
    model = get_model_timeSeries(config, seq_len, vocab_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9, weight_decay=1e-4)

    total_steps = config["num_epochs"] * (config["train_count"] // config["batch_size"])
    warmup_steps = int(0.05 * total_steps)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_steps
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-6
        )
    ],
    milestones=[warmup_steps]
)


    initial_epoch = 0
    global_step = 0

    # load latest model and start from there
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename, map_location=torch.device('cpu'))
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    #recalculating original numbers
    i2v_dict = index_to_value_dict(config["vocab_size"], config["extra_tokens"])
    i2v = torch.zeros(config["vocab_size"] + len(config["extra_tokens"]) + 1).to(device)
    for k, v in i2v_dict.items():
        if(v == "ukn"):
            i2v[int(k)] = 0.0
        else:
            i2v[int(k)] = float(v)

    
    #loss function
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"]).to(device)
    loss_grad = nn.MSELoss().to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["noisy_TimeSeries"].to(device)                                #(Batch,seq_len) --> index shape
            encoder_input_removed = batch["interpolation_noisy_TimeSeries"].to(device)          #(Batch,seq_len) --> index shape
            decoder_input = batch["groundTruth"].to(device)                                     #(Batch,seq_len) --> float shape
            div_term = batch["div_term"].to(device)                                             #(Batch) --> float shape
            min_value = batch["min_value"].to(device)                                           #(Batch) --> float shape
            noise_std = batch["noise_std"]                                                    #(Batch) --> float shape
            time = torch.linspace(0, 1, steps=1000).unsqueeze(0).to(device)


            #apply model
            if(config["remove_parts"]):
                #train model with interpolation purpose
                encoder_output = model.encode(encoder_input_removed, None, time)  #(Batch, seq_len) --> (Batch, seq_len, d_model)
            else:
                #train model without interpolation purpose
                encoder_output = model.encode(encoder_input, None, time)          #(Batch, seq_len) --> (Batch, seq_len, d_model)

            proj_output = model.project(encoder_output)                     #(Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)

            proj_output_copy = proj_output[0,:,:]       #use first batch entry to store results
            _, indices = torch.max(proj_output_copy, 1) #calculate highest value with equivalent index in each row


            #create copies to store training results
            proj_output_copy = indices[:]                   #prediction:    index
            decoder_input_copy = decoder_input[0,:]         #groundTruth:   normalized
            noise_copy = encoder_input[0,:]                 #noise:         index
            noise_removed_copy = encoder_input_removed[0,:] #noise removed: index
            div_term_copy = div_term[0]                     #div_term:      float
            min_value_copy = min_value[0]                   #min_value:     float
            noise_std_copy = noise_std[0]                   #noise_std:     float


            groundTruth = batch["groundTruth_indices"].to(device).view(-1)  #(batch,seq_len) --> (batch * seq_len)
            prediction = proj_output.view(-1, vocab_size)                   #(batch,seq_len, 1) --> (batch * seq_len, tgt_vocab_size)
            lossCE = loss_fn(prediction, groundTruth)                         #calculate cross-entropy-loss
            

            probs = torch.softmax(proj_output, dim=-1)     # (B,S,V)

            # i2v_values: (V,) oder (V,1) als float tensor auf device
            # Erwartungswert: pred_value[b,s] = sum_v probs[b,s,v] * i2v_values[v]
            pred_value = (probs * i2v.view(1,1,-1)).sum(dim=-1)   # (B,S)

            pred_value = pred_value * div_term.unsqueeze(-1) + min_value.unsqueeze(-1)
            prediction_grad = pred_value[:, 1:] - pred_value[:, :-1]

            groundTruth = batch["groundTruth"].to(device)
            groundTruth = groundTruth * div_term.unsqueeze(-1) + min_value.unsqueeze(-1)
            groundTruth_grad = groundTruth[:,1:] - groundTruth[:,:-1]

            lossGradient = loss_grad(prediction_grad, groundTruth_grad)

            g_ce   = grad_norm(lossCE, model)
            g_grad = grad_norm(lossGradient, model)

            alpha = (g_ce / (g_grad + 1e-8)).detach()

            loss = lossCE + alpha * lossGradient
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.5f}; lossCE: {lossCE.item():6.3f}; lossGrad: {lossGradient.item():6.3f}"})

            #backpropagate the loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            #update the weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1



        if epoch % 20 == 0:
            #store training results
            df = pd.DataFrame()
            df.loc[:,f"noise"] = (noise_copy).detach().cpu().numpy()                                                #index form
            df.loc[:,f"noise_removed"] = (noise_removed_copy).detach().cpu().numpy()                                #index form
            df.loc[:,f"groundTruth"] = ((decoder_input_copy*div_term_copy)+min_value_copy).detach().cpu().numpy()   #float form
            df.loc[:,f"prediction"] = (proj_output_copy).detach().cpu().numpy()                                     #index form
            df.loc[0,f"min_value"] = (min_value_copy).detach().cpu().numpy()                                        #float form
            df.loc[0,f"div_term"] = (div_term_copy).detach().cpu().numpy()                                          #float form
            df.loc[0,f"noise_std"] = (noise_std_copy).detach().cpu().numpy()                                        #float form
            df.to_csv(f"results_train/train_epoch_{epoch}.csv", index=False)

            # Run validation at the end of every epoch
            run_validation_TimeSeries(model, val_dataloader, device, 4, config, epoch)
            
            #save the model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step
            }, model_filename)
   

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model_TimeSeries_paper(config)