import torch 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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

def get_model_timeSeries(config, seq_len, vocab_size_src):
    model = build_encoder_interpolation_uknToken_projection(vocab_size_src, seq_len, config["d_model"], dropout=config["dropout"])
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
            df.loc[:,f"prediction_mu_{count}"] = model_out[0,:,0].cpu().numpy()                    #index form
            df.loc[:,f"prediction_sigma_{count}"] = model_out[0,:,1].cpu().numpy()                    #index form
            df.loc[0,f"min_value_{count}"] = min_value[0].cpu().numpy()                     #float form
            df.loc[0,f"div_term_{count}"] = div_term[0].cpu().numpy()                       #float form
            df.loc[0,f"noise_std_{count}"] = noise_std[0].cpu().numpy()                     #float form
            count +=1
            if count == num_examples:
                df.to_csv(f"results_val/val_epoch_{epoch_nr}.csv", index=False)
                break
        
        # gespeicherte Zeitreihen laden und auswerten lassen und heatmap abspeichern
        # erst ohne Maske
        df = pd.read_csv(f"results_val/heat_map_data.csv")
        encoder_input = torch.tensor(df["noisy_TimeSeries"]).unsqueeze(0).to(device) 
        proj_output = greedy_decode_timeSeries_paper(model, encoder_input, None)
        prob_distribution = np.memmap(f"results_train/prob_distribution_epoch_{epoch_nr}.npy", dtype='float32', mode='w+', shape=proj_output[0,:,:].shape)
        prob_distribution[:] = proj_output[0,:,:].detach().cpu().numpy()    #(seq_len, vocab_size)
        prob_distribution.flush()

        # dann mit Maske 
        encoder_input_removed = torch.tensor(df["noisy_TimeSeries_removed"]).unsqueeze(0).to(device) 
        proj_output = greedy_decode_timeSeries_paper(model, encoder_input_removed, None)
        prob_distribution = np.memmap(f"results_train/prob_distribution_epoch_masking_{epoch_nr}.npy", dtype='float32', mode='w+', shape=proj_output[0,:,:].shape)
        prob_distribution[:] = proj_output[0,:,:].detach().cpu().numpy()    #(seq_len, vocab_size)
        prob_distribution.flush()


def greedy_decode_timeSeries_paper(model, source: torch.Tensor, time: torch.Tensor):
    
    encoder_output = model.encode(source, None, time)
    
    proj_out = model.project(encoder_output)           #(batch,seq_len, d_model) -> (bach,seq_len, 2)

    # _, indices = torch.max(proj_out, dim=1)                 #(batch, seq_len, vocab_size) -> (seq_len)
    
    return proj_out    #(batch, seq_len, 2)


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


def wasserstein1_cdf_loss(p: torch.Tensor, q: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    p, q: [B, L] Wahrscheinlichkeiten (sum=1 pro Zeile), >=0
    """
    cdf_p = torch.cumsum(p, dim=-1)
    cdf_q = torch.cumsum(q, dim=-1)
    w1 = torch.sum(torch.abs(cdf_p - cdf_q), dim=-1)  # [B]

    if reduction == "mean":
        return w1.mean()
    if reduction == "sum":
        return w1.sum()
    return w1  # 'none



def train_model_TimeSeries_paper(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    os.makedirs("weights/", exist_ok=True)
    os.makedirs("results_train", exist_ok=True)
    os.makedirs("results_val", exist_ok=True)
    
    train_dataloader, val_dataloader, seq_len, vocab_size_src = get_ds_timeSeries(config)
    model = get_model_timeSeries(config, seq_len, vocab_size_src).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9, weight_decay=1e-4)

    train_count = config["train_count"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]

    total_steps = num_epochs * (train_count // batch_size)
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
            eta_min=1e-5
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
        scheduler.load_state_dict(state['schedulaer_state_dict'])

    #recalculating original numbers
    # i2v_dict = index_to_value_dict(config["vocab_size"])
    # i2v = torch.zeros(config["vocab_size"] + 1).to(device)
    # for k, v in i2v_dict.items():
    #     print(f"key: {k}; value: {v}")
    #     i2v[int(k)] = float(v)

    #tracking loss
    writer = SummaryWriter("results_train/my_experiment")
    counter = 0

    #loss weighting
    # loss_w1_weight = config["loss_w1"]
    # loss_entropy_penalty_weight = config["loss_entropy_penalty"]
    # loss_curv_weight = config["loss_curv"]
    
    #loss function
    # loss_fn = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"]).to(device)
    # loss_grad = nn.MSELoss().to(device)
    # soft_argmax = nn.SmoothL1Loss().to(device)

    for epoch in range(initial_epoch, num_epochs):
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
            # time = torch.linspace(0, 1, steps=1000).unsqueeze(0).to(device)


            #apply model
            if(config["remove_parts"]):
                #train model with interpolation purpose
                encoder_output = model.encode(encoder_input_removed, None, None)  #(Batch, seq_len) --> (Batch, seq_len, d_model)
            else:
                #train model without interpolation purpose
                encoder_output = model.encode(encoder_input, None, None)          #(Batch, seq_len) --> (Batch, seq_len, d_model)

            proj_output = model.project(encoder_output)                     #(Batch, seq_len, d_model) --> (Batch, seq_len, 2)

            proj_output_copy = proj_output[0,:,:]       #use first batch entry to store results
            # _, indices = torch.max(proj_output_copy, 1) #calculate highest value with equivalent index in each row


            #create copies to store training results
            proj_output_copy_mu = proj_output_copy[:, 0]    #prediction mu
            proj_output_copy_std = proj_output_copy[:, 1]   #prediction std
            decoder_input_copy = decoder_input[0,:]         #groundTruth:   normalized
            noise_copy = encoder_input[0,:]                 #noise:         index
            noise_removed_copy = encoder_input_removed[0,:] #noise removed: index
            div_term_copy = div_term[0]                     #div_term:      float
            min_value_copy = min_value[0]                   #min_value:     float
            noise_std_copy = noise_std[0]                   #noise_std:     float

            #was muss ich machen?
            #nehme groundtruth und redimensioniere zu (batch*seq_len, 1)
            groundTruth = batch["groundTruth_indices"].to(device)
            groundTruth = groundTruth.view(groundTruth.shape[0]*groundTruth.shape[1],1) #(B,S) --> (B*S)
            #erstelle vocab tokens als ein Tensor
            tokens = torch.ones_like(groundTruth).to(device) * torch.arange(config["vocab_size"] + 1, device=device).float().unsqueeze(0) #(B*S,V)
            #aus mu und std für jeden eintrag von prediction eine gauss verteilung berechnen
            prediction = proj_output.view(-1, 2)                   #(batch,seq_len, 2) --> (batch * seq_len, 2)
            prediction_mu = prediction[:,0].unsqueeze(-1)         #(B*S, 1)
            prediction_std = prediction[:,1].unsqueeze(-1)        #(B*S, 1)
            prediction_prob =  1/(prediction_std*np.sqrt(2*np.pi)) * torch.exp(-0.5 * ((tokens - prediction_mu) / (prediction_std)) ** 2) #(B*S,V)

            #groundtruth probability bestimmen
            std_factor = 0.6
            std = (prediction_std * std_factor)
            # std = (max(10, std.item()))
            groundTruth_extended = groundTruth * torch.ones(1, config["vocab_size"]+1, device=device).float()
            groundTruth_prob =  1/(std*np.sqrt(2*np.pi)) * torch.exp(-0.5 * ((tokens - groundTruth_extended) / (std)) ** 2)

            
            # prediction = proj_output.view(-1, vocab_size_tgt)                   #(batch,seq_len, 1) --> (batch * seq_len, tgt_vocab_size)
            # prediction_prob = torch.softmax(prediction, dim=-1)    #(B*S, V)
            # prediction_prob_mean = (prediction_prob*tokens).sum(dim=-1)  #(B*S)
            # prediction_prob_std = torch.sqrt((prediction_prob*(tokens - prediction_prob_mean.unsqueeze(-1))**2).sum(dim=-1)).mean()
            

            #calculate gauss ce loss
            std_factor = 0.6
            std = (std * std_factor)
            groundTruth_extended = groundTruth * torch.ones(1, config["vocab_size"]+1, device=device).float()
            groundTruth_prob =  1/(std*np.sqrt(2*np.pi)) * torch.exp(-0.5 * ((tokens - groundTruth_extended) / (std)) ** 2)
            log_p = F.log_softmax(prediction, dim=-1)
            loss_gauss_ce = -(groundTruth_prob * log_p).sum(dim=-1).mean()
            # std = (max(10, std.item()))

            #gaussian distribution around the groundtruth token
            # x = torch.ones_like(groundTruth).to(device) * torch.arange(config["vocab_size"] + 1, device=device).float().unsqueeze(0)
            # groundTruth_extended = groundTruth * torch.ones(1, config["vocab_size"]+1, device=device).float()
            # groundTruth_prob =  1/(std*np.sqrt(2*np.pi)) * torch.exp(-0.5 * ((tokens - groundTruth_extended) / (std)) ** 2)
            # log_p = F.log_softmax(prediction, dim=-1)
            # loss_gauss_ce = -(groundTruth_prob * log_p).sum(dim=-1).mean()

            #calculate soft argmax
            # prediction_prob = torch.softmax(prediction, dim=-1) #(B,L,V)
            # vocab_tokens = torch.arange(config["vocab_size"]+1).to(device) #(V)
            # prediction_token = (prediction_prob*vocab_tokens).sum(dim=-1).view(-1) #(B*L)
            # loss_soft_argmax = ((prediction_token - groundTruth.float())**2).mean()

            #wasserstein loss #(B,L,V)
            loss_w1 = wasserstein1_cdf_loss(prediction_prob, groundTruth_prob, reduction="mean")
            


            #calculate entropy - penalty
            loss_entropy_penalty = - (prediction_prob * torch.log(prediction_prob + 1e-8)).sum(dim=-1).mean()


            #calculate KL-loss
            prediction_prob_log = torch.log(prediction_prob + 1e-8)
            loss_kl = F.kl_div(prediction_prob_log, groundTruth_prob, reduction="batchmean")
            

            #calculate curvature loss (2. difference)
            # pred_norm = (prediction_prob * i2v.view(1,1,-1)).sum(dim=-1)   # (B,S)
            # d2 = pred_norm[:,2:] - 2*pred_norm[:,1:-1] + pred_norm[:,:-2]
            # loss_curv = torch.sqrt((d2**2 + (1e-3)**2)).mean()

            # eps = 1e-8
            # #total loss
            # start_epochs = 20
            # if(epoch < start_epochs):
            #     loss = loss_gauss_ce + loss_w1_weight * grad_gauss_ce / (grad_w1 + eps) * loss_w1
            # else:
            #     loss = loss_gauss_ce + loss_w1_weight * grad_gauss_ce / (grad_w1 + eps) * loss_w1 + loss_entropy_penalty_weight * grad_gauss_ce / (grad_entropy_penalty + eps) * loss_entropy_penalty

            #     if counter < (train_count // batch_size) * (80):
            #         loss_w1_weight += (1-config["loss_w1"]) / ((train_count // batch_size) * (80))
            #         loss_entropy_penalty_weight += (1-config["loss_entropy_penalty"]) / ((train_count // batch_size) * (80))

            #total loss
            loss = loss_kl
            # start_epochs = 15
            # middle_epoch = 30
            # if(epoch < start_epochs):
            #     loss = loss_gauss_ce + loss_w1_weight * loss_w1
            # elif(epoch < middle_epoch):
            #     loss = loss_gauss_ce + loss_w1_weight * loss_w1 + loss_entropy_penalty_weight * loss_entropy_penalty
            # else:
            #     loss = loss_gauss_ce + loss_w1_weight * loss_w1 + loss_entropy_penalty_weight * loss_entropy_penalty + loss_curv_weight * loss_curv

                # if counter < (train_count // batch_size) * (80):
                #     loss_w1_weight += (1-config["loss_w1"]) / ((train_count // batch_size) * (80))
                #     loss_entropy_penalty_weight += (1-config["loss_entropy_penalty"]) / ((train_count // batch_size) * (80))


            #log the loss values
            writer.add_scalar('loss/total_loss', loss.item(), counter)
            writer.add_scalar('loss/kl_loss', loss_kl.item(), counter)
            # writer.add_scalar('loss/gauss_ce_loss', loss_gauss_ce.item(), counter)
            # writer.add_scalar('loss/w1', loss_w1.item(), counter)
            # writer.add_scalar('loss/loss_entropy_penalty', loss_entropy_penalty.item(), counter)

            # if counter % 1000 == 0:
                #calculate gradient norms
                # grad_gauss_ce = grad_norm(loss_gauss_ce, model)
                # grad_w1 = grad_norm(loss_w1, model)
                # grad_entropy_penalty = grad_norm(loss_entropy_penalty, model)
                # grad_loss = grad_norm(loss, model)
                #log gradient norms
                # writer.add_scalar('grad_norm/gauss_ce', grad_gauss_ce.item(), counter)
                # writer.add_scalar('grad_norm/w1', grad_w1.item(), counter)
                # writer.add_scalar('grad_norm/entropy_penalty', grad_entropy_penalty.item(), counter)
                # writer.flush()

            batch_iterator.set_postfix({
            "loss": f"{loss.item():6.2f}",
            "loss_kl": f"{loss_kl.item():6.2f}",
            "gauss_ce": f"{loss_gauss_ce.item():6.2f}",
            "w1": f"{loss_w1.item():6.2f}",
            "entropy_pen": f"{loss_entropy_penalty.item():6.2f}",
            # "loss_curv": f"{loss_curv.item():6.2f}",
            # "target_std": f"{std:6.2f}",
            # "grad_loss": f"{grad_loss.item():6.2f}",
            # "gradd_gauss_ce": f"{grad_gauss_ce.item():6.2f}",
            # "grad_w1": f"{grad_w1.item():6.2f}"
            # "grad_entropy_pen": f"{grad_entropy_penalty.item():6.2f}"
            })

            #backpropagate the loss
            loss.backward()


            #update the weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            counter += 1



        if epoch % 20 == 0:
            #store training results
            df = pd.DataFrame()
            df.loc[:,f"noise"] = (noise_copy).detach().cpu().numpy()                                                #index form
            df.loc[:,f"noise_removed"] = (noise_removed_copy).detach().cpu().numpy()                                #index form
            df.loc[:,f"groundTruth"] = ((decoder_input_copy*div_term_copy)+min_value_copy).detach().cpu().numpy()   #float form
            df.loc[:,f"prediction_mu"] = (proj_output_copy_mu).detach().cpu().numpy()                                     #index form
            df.loc[:,f"prediction_std"] = (proj_output_copy_std).detach().cpu().numpy()                                     #index form
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
                "schedulaer_state_dict": scheduler.state_dict(),
                "global_step": global_step
            }, model_filename)
   

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model_TimeSeries_paper(config)