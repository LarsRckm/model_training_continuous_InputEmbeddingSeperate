import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.widgets import CheckButtons
import torch
import math
from config import get_config
from model import build_encoder_interpolation_uknToken_projection
from train_model_encoder_interpolation_CE import greedy_decode_timeSeries_paper as greedy_decode_timeSeries_paper_projection
from useful import value_to_index_dict, index_to_value_dict, round_numbers_individually, calc_exp, round_with_exp
from pathlib import Path

#
class DraggableCheckButtons:
    def __init__(self, check_buttons, ax, width, height):
        self.check_buttons = check_buttons
        self.ax = ax
        self.press = None
        self.background = None
        self.width = width
        self.height = height

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cidscroll = self.ax.figure.canvas.mpl_connect('scroll_event', self.on_scroll)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.ax: return

        # contains, attrd = self.check_buttons.ax.contains(event)
        # if not contains: return
        # Überprüfen, ob ein Werkzeug aktiviert ist
        if self.ax.figure.canvas.manager.toolbar.mode != '':
            return

        x0, y0 = self.check_buttons.ax.get_position().bounds[:2]
        self.press = x0, y0, event.x, event.y

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.ax: return

        if self.ax.figure.canvas.manager.toolbar.mode != '':
            return

        x0, y0, xpress, ypress = self.press
        dx = event.x - xpress
        dy = event.y - ypress

        # Konvertierung von Pixeln in normierte Einheiten (Zoll)
        new_x = x0 + dx / self.ax.figure.dpi
        new_y = y0 + dy / self.ax.figure.dpi
        # / self.ax.figure.dpi

        # Neue Position setzen
        self.check_buttons.ax.set_position([new_x, new_y, self.width, self.height])
        self.ax.figure.canvas.draw()

    def on_release(self, event):
        'on release we reset the press data'
        self.press = None
        self.ax.figure.canvas.draw()
    
    def on_scroll(self, event):
        'reset the press data on scroll'
        self.press = None

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.ax.figure.canvas.mpl_disconnect(self.cidpress)
        self.ax.figure.canvas.mpl_disconnect(self.cidrelease)
        self.ax.figure.canvas.mpl_disconnect(self.cidmotion)
        self.ax.figure.canvas.mpl_disconnect(self.cidscroll)

#needed
def predict_encoder_interpolation_projection_roundedInput(seq_length: int, model_filename, y_noisy_spline,min_value_spline, max_value_spline, config, mask):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab_size = config["vocab_size"]
    vocab_extra_tokens = config["extra_tokens"]
    v2i = value_to_index_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    i2v = index_to_value_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    exp = calc_exp(smallest_number=1/vocab_size)

    model = build_encoder_interpolation_uknToken_projection(len(v2i), seq_length)


    print(f'Preloading model {model_filename}')
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    model.eval()    


    with torch.no_grad():
        #min-max scaling
        div_term = max_value_spline-min_value_spline
        encoder_input_removed = torch.tensor((y_noisy_spline-min_value_spline)/div_term,dtype=torch.float32)[:].to(device)
        
        #masking       
        mask_indices = np.where(mask == 1)[0] # mask == 1 --> remove
        
        #round scaled input timeseries taking into account vocab_size
        encoder_input_removed = round_numbers_individually(vocab_size,encoder_input_removed)[:]

        #replace discrete values with indices
        encoder_input_removed[mask_indices] = v2i["ukn"] #replace interpolation points with the index-equivalent of the "ukn" token
        encoder_input_removed = encoder_input_removed.apply_(lambda x: x if x>vocab_size else v2i[f"{round_with_exp(x, exp)}"]).type(torch.long) #map discrete values to indices

        #create prediction
        model_out = greedy_decode_timeSeries_paper_projection(model, encoder_input_removed.unsqueeze(0))

    #map indices to discrete values
    model_out = model_out.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])
    encoder_input_removed = encoder_input_removed.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])
    
    #denormalize discrete values with stores values of min-max scaling
    model_out = (model_out*div_term)+min_value_spline
    encoder_input_removed = (encoder_input_removed*div_term)+min_value_spline


    return model_out, encoder_input_removed

#needed
def plot_multiple_pred_with_names_error_bar(x_values, y_values_prediction_total, titles, labels):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    labelSize = 20
    labelSizeLegend = 18
    lineWidth = 4
    width = 0.15
    height = 0.15
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True)

    for i in range(length):
        axis[i].tick_params(axis='both', which='both', labelsize=labelSize)
        axis[i].set_title(titles[i],fontsize=labelSize,fontweight='bold')
        axis[i].set_ylabel(labels[i],fontsize=labelSize,fontweight='bold')
        axis[i].margins(x=0.01)
    lines = []
    min_value = None
    max_value = None
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            else:
                if j > num_plots-4:
                    lower_border = plot[0]
                    upper_border = plot[1]
                    color = plot[2]
                    axis[i].fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
                else:
                    y_values = plot[0]
                    name = plot[1]
                    color = plot[2]
                    line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                    min_value = min(y_values)
                    max_value = max(y_values)
    
    #calculate interpolation intervals from 


    axis[0].legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    offset = (max_value-min_value)*0.05
    axis[1].set_ylim(min_value-offset, max_value+offset)
    
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize)#deutsch

    
    plt.show()

#needed
def plot_multiple_pred_with_names_error_bar_interpolationPainting(x_values, y_values_prediction_total, titles, labels, intervals, title, folderName):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''

    # DIN A4 Breite in Zoll
    a4_width_inch = 8.27/2  # für volle Breite

    # DPI des Bildes (300 für Veröffentlichungen)
    dpi = 300

    # Schriftgröße in LaTeX-Punkten (z. B. 10pt, 12pt)
    latex_font_size_pt = 10

    # Umrechnung von Punkt in Zoll (1 pt = 1/72 Zoll)
    points_to_inches = 1 / 72

    # Umrechnung der Schriftgröße in Zoll
    latex_font_size_inch = latex_font_size_pt * points_to_inches


    labelSize = latex_font_size_pt
    labelSizeLegend = latex_font_size_pt
    lineWidth = 3
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True, figsize=(a4_width_inch, a4_width_inch / 1.618))

    for i in range(length):
        axis[i].tick_params(axis='both', which='both', labelsize=labelSize)
        # axis[i].set_title(titles[i],fontsize=labelSize,fontweight='bold')
        axis[i].set_ylabel(labels[i],fontsize=labelSize)
        axis[i].margins(x=0.0)
    lines = []
    min_value = None
    max_value = None
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            else:
                if j > num_plots-4:
                    lower_border = plot[0]
                    upper_border = plot[1]
                    color = plot[2]
                    axis[i].fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
                else:
                    y_values = plot[0]
                    name = plot[1]
                    color = plot[2]
                    line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                    min_value = min(y_values)
                    max_value = max(y_values)
    
    #calculate interpolation intervals from 
    for bottom_border, upper_border in intervals:
        axis[0].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.15)
        axis[1].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.15)

    # axis[0].legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    axis[0].legend(loc='best',prop={'size': labelSizeLegend})
    offset = (max_value-min_value)*0.05
    axis[1].set_ylim(min_value-offset, max_value+offset)
    
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize)#deutsch

    
    fmt = "svg"
    save_path = Path(f"save_figure/{folderName}/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
    plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.show()


def plot_multiple_pred_with_names_error_bar_interpolationPainting_PAPER(x_values, y_values_prediction_total, titles, labels, intervals, title, folderName):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''

    # DIN A4 Breite in Zoll
    a4_width_inch = 8.27/2  # für volle Breite

    # DPI des Bildes (300 für Veröffentlichungen)
    dpi = 300

    # Schriftgröße in LaTeX-Punkten (z. B. 10pt, 12pt)
    latex_font_size_pt = 10

    # Umrechnung von Punkt in Zoll (1 pt = 1/72 Zoll)
    points_to_inches = 1 / 72

    # Umrechnung der Schriftgröße in Zoll
    latex_font_size_inch = latex_font_size_pt * points_to_inches


    labelSize = latex_font_size_pt
    labelSizeLegend = latex_font_size_pt
    lineWidth = 3
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True, figsize=(a4_width_inch, a4_width_inch / 1.618))

    for ax in axis:
        ax.tick_params(axis='both', which='both', labelsize=labelSize)
        # axis[i].set_title(titles[i],fontsize=labelSize,fontweight='bold')
        ax.set_ylabel(labels[i],fontsize=labelSize)
        ax.margins(x=0.0)
        
    lines = []
    min_value = None
    max_value = None
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            else:
                if j > num_plots-4:
                    lower_border = plot[0]
                    upper_border = plot[1]
                    color = plot[2]
                    axis[i].fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
                else:
                    y_values = plot[0]
                    name = plot[1]
                    color = plot[2]
                    line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                    min_value = min(y_values)
                    max_value = max(y_values)
    
    #calculate interpolation intervals from 
    for bottom_border, upper_border in intervals:
        axis[0].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.15)
        axis[1].axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.15)

    # axis[0].legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    axis[0].legend(loc='best',prop={'size': labelSizeLegend})
    offset = (max_value-min_value)*0.05
    axis[1].set_ylim(min_value-offset, max_value+offset)
    
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize)#deutsch

    
    fmt = "svg"
    save_path = Path(f"save_figure/{folderName}/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
    plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
    # plt.show()


#needed
def plot_multiple_pred_with_names_error_bar_single(x_values, y_values_prediction_total, titles, labels, intervals):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    labelSize = 20
    labelSizeLegend = 18
    # lineWidth = 7
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True, figsize=(16,9))

    for i in range(length):
        axis.tick_params(axis='both', which='both', labelsize=labelSize)
        axis.set_title(titles[i],fontsize=labelSize,fontweigt='bold')
        axis.set_ylabel(labels[i],fontsize=labelSize,fontweight='bold')
        axis.margins(x=0.01)
    lines = []
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                lineWidth = plot[3]
                line, = axis.plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            # else:
            #     if j > num_plots-4:
            #         lower_border = plot[0]
            #         upper_border = plot[1]
            #         color = plot[2]
            #         axis.fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
            #     else:
            #         y_values = plot[0]
            #         name = plot[1]
            #         color = plot[2]
            #         line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
            #         min_value = min(y_values)
            #         max_value = max(y_values)


    axis.legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    #calculate interpolation intervals from 
    for bottom_border, upper_border in intervals:
        axis.axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.15)
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize,fontweight='bold') #deutsch

    
    # plt.subplots_adjust(right=0.85)
    # rax = plt.axes([0.4, 0.05, width, height])
    # lines_by_label = {l.get_label(): l for l in lines}
    # line_colors = [l.get_color() for l in lines]
    # check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props={'color': line_colors},
    # frame_props={'edgecolor': line_colors},
    # check_props={'facecolor': line_colors},)
    # for label in check.labels:
    #     label.set_fontsize(labelSize)
    # def callback(label):
    #         ln = lines_by_label[label]
    #         ln.set_visible(not ln.get_visible())
    #         ln.figure.canvas.draw_idle()
    # check.on_clicked(callback)
    # draggable_check = DraggableCheckButtons(check, axis[0], width, height)
    # draggable_check.connect()

    
    plt.show()

#needed
def plot_multiple_pred_with_names_error_bar_single_presentation(x_values, y_values_prediction_total, titles, labels, intervalle):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    labelSize = 20
    labelSizeLegend = 18
    lineWidth = 6
    width = 0.15
    height = 0.15
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True)

    for i in range(length):
        axis.tick_params(axis='both', which='both', labelsize=labelSize)
        axis.set_title(titles[i],fontsize=labelSize,fontweight='bold')
        axis.set_ylabel(labels[i],fontsize=labelSize,fontweight='bold')
        axis.margins(x=0.01)
    lines = []
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                line, = axis.plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            # else:
            #     if j > num_plots-4:
            #         lower_border = plot[0]
            #         upper_border = plot[1]
            #         color = plot[2]
            #         axis.fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
            #     else:
            #         y_values = plot[0]
            #         name = plot[1]
            #         color = plot[2]
            #         line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
            #         min_value = min(y_values)
            #         max_value = max(y_values)


    # axis.legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize,fontweight='bold') #deutsch

    #calculate interpolation intervals from 
    for bottom_border, upper_border in intervalle:
        axis.axvspan(bottom_border - 0.5, upper_border + 0.5, color='red', alpha=0.05)
    
    # plt.subplots_adjust(right=0.85)
    rax = plt.axes([0.4, 0.05, width, height])
    lines_by_label = {l.get_label(): l for l in lines}
    line_colors = [l.get_color() for l in lines]
    check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props={'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},)
    for label in check.labels:
        label.set_fontsize(labelSize)
    def callback(label):
            ln = lines_by_label[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()
    check.on_clicked(callback)
    draggable_check = DraggableCheckButtons(check, axis, width, height)
    draggable_check.connect()

    
    plt.show()

#needed
def plot_multiple_pred_with_names_error_bar_single_presentation_02(x_values, y_values_prediction_total, titles, labels):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    labelSize = 20
    labelSizeLegend = 18
    lineWidth = 6
    width = 0.15
    height = 0.15
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True)

    for i in range(length):
        axis.tick_params(axis='both', which='both', labelsize=labelSize)
        axis.set_title(titles[i],fontsize=labelSize,fontweight='bold')
        axis.set_ylabel(labels[i],fontsize=labelSize,fontweight='bold')
        axis.margins(x=0.01)
    lines = []
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        num_plots = len(y_values_copy)
        for j in range(num_plots):
            plot = y_values_copy[j]
            if i==0:
                y_values = plot[0]
                name = plot[1]
                color = plot[2]
                line, = axis.plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
                lines.append(line)
            # else:
            #     if j > num_plots-4:
            #         lower_border = plot[0]
            #         upper_border = plot[1]
            #         color = plot[2]
            #         axis.fill_between(x_values, lower_border, upper_border, color=color, alpha=0.3)
            #     else:
            #         y_values = plot[0]
            #         name = plot[1]
            #         color = plot[2]
            #         line, = axis[i].plot(x_values, y_values, label=f"{name}",color=color,linewidth=lineWidth)
            #         min_value = min(y_values)
            #         max_value = max(y_values)


    # axis.legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    
    # plt.xlabel("Time Steps",fontsize=labelSize)#englisch
    plt.xlabel("Zeitschritte",fontsize=labelSize,fontweight='bold') #deutsch

    
    # plt.subplots_adjust(right=0.85)
    rax = plt.axes([0.4, 0.05, width, height])
    lines_by_label = {l.get_label(): l for l in lines}
    line_colors = [l.get_color() for l in lines]
    check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props={'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},)
    for label in check.labels:
        label.set_fontsize(labelSize)
    def callback(label):
            ln = lines_by_label[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()
    check.on_clicked(callback)
    draggable_check = DraggableCheckButtons(check, axis, width, height)
    draggable_check.connect()

    
    plt.show()

#needed
def denoise_floaterCurrent_encoder_interpolation_CE(latest_encoderModel_filename, seq_len, current, config, mask=None):
    print("function called")
    shape_current = current.shape[0]

    prediction_encoder = torch.tensor([])
    encoder_input_removed_tensor = torch.tensor([])

    i=0
    exp = calc_exp(smallest_number=1/config["vocab_size"])
    predict_len = 800
    ueberlappen_len = 1000-predict_len


    while i < shape_current:
        #first prediction includes first 1000 data points
        if i == 0:
            lower_border = i
            upper_border = i+1000

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_copy = mask[lower_border:upper_border]
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy = current[lower_border:upper_border]
            current_copy_min_max = current_copy[mask_indices_in] #extract all values that are not masked out for minimum and maximum calculation
            
            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"min:{min(current_copy)}")
            print(f"max:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed), 0)

            i += 1000
        
        #following predictions use 500 data points of the previous prediction and 500 new data points
        #the rear part of the prediction (last 500 data points of prediction) is appended to prediction array
        else:
            lower_border = i
            upper_border = i+predict_len
            prediction = prediction_encoder[-ueberlappen_len:]                                  #past 500 values of total prediction array so far
            current_copy = current[lower_border:upper_border]                       #next 500 values of noisy data after end of prediction
            current_copy = torch.cat((prediction, torch.tensor(current_copy)), 0)   #connect previous tensors to one 1000 values tensor

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_previous_prediction = np.zeros(ueberlappen_len, dtype=int) #all 500 values from previous prediction are taken into account for next prediction -> mask index = 0
                mask_copy = mask[lower_border:upper_border]         #take mask values for the next 500 values
                mask_copy = np.concatenate((mask_previous_prediction, mask_copy), 0) #connect the masking values
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy_min_max = current_copy[mask_indices_in]    #extract all values that are not masked out for minimum and maximum calculation

            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"min:{min(current_copy)}")
            print(f"max:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction[ueberlappen_len:]),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed[ueberlappen_len:]), 0)

            i += predict_len



    prediction_encoder = prediction_encoder.detach().numpy()
    prediction_encoder_sliding = sliding_window(prediction_encoder, 2) 
    #apply two-side sliding window
    # prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, 4000)   #apply two-side sliding window again to smooth more
    # print("in Sliding")
    for i in range(5):
        prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, 2)

    return prediction_encoder,prediction_encoder_sliding, encoder_input_removed_tensor




def denoise_floaterCurrent_encoder_interpolation_CE_forRawDVA(latest_encoderModel_filename, seq_len, current, config, windowSize, windowIterations, mask=None):
    shape_current = current.shape[0]

    prediction_encoder = torch.tensor([])
    encoder_input_removed_tensor = torch.tensor([])

    i=0
    exp = calc_exp(smallest_number=1/config["vocab_size"])
    predict_len = 800
    ueberlappen_len = 1000-predict_len


    while i < shape_current:
        #first prediction includes first 1000 data points
        if i == 0:
            lower_border = i
            upper_border = i+1000

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_copy = mask[lower_border:upper_border]
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy = current[lower_border:upper_border]
            current_copy_min_max = current_copy[mask_indices_in] #extract all values that are not masked out for minimum and maximum calculation
            
            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"\tmin:{min(current_copy)}")
            print(f"\tmax:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed), 0)

            i += 1000
        
        #following predictions use 500 data points of the previous prediction and 500 new data points
        #the rear part of the prediction (last 500 data points of prediction) is appended to prediction array
        else:
            lower_border = i
            upper_border = i+predict_len
            prediction = prediction_encoder[-ueberlappen_len:]                                  #past 500 values of total prediction array so far
            current_copy = current[lower_border:upper_border]                       #next 500 values of noisy data after end of prediction
            current_copy = torch.cat((prediction, torch.tensor(current_copy)), 0)   #connect previous tensors to one 1000 values tensor

            #assert upper_border not larger than total timeseries length
            if upper_border > shape_current:
                upper_border = shape_current

            if type(mask) != None:
                mask_previous_prediction = np.zeros(ueberlappen_len, dtype=int) #all 500 values from previous prediction are taken into account for next prediction -> mask index = 0
                mask_copy = mask[lower_border:upper_border]         #take mask values for the next 500 values
                mask_copy = np.concatenate((mask_previous_prediction, mask_copy), 0) #connect the masking values
                mask_indices_in = np.where(mask_copy == 0)[0]   #values which are not masked out
                mask_indices_out = np.where(mask_copy == 1)[0]  #values which are masked out

            current_copy_min_max = current_copy[mask_indices_in]    #extract all values that are not masked out for minimum and maximum calculation

            #calculate global min and max values
            max_value_current = math.ceil(max((current_copy_min_max))*(10**exp))/(10**exp)
            min_value_current = math.floor(min((current_copy_min_max))*(10**exp))/(10**exp)
            print(f"\tmin:{min(current_copy)}")
            print(f"\tmax:{max(current_copy)}")    

            #create prediction
            prediction, encoder_input_removed = predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_len,latest_encoderModel_filename,current_copy,min_value_current, max_value_current,mask_indices_out, config)
            prediction = prediction.squeeze()

            #append results to existing torch tensors
            prediction_encoder = torch.cat((prediction_encoder,prediction[ueberlappen_len:]),0)
            encoder_input_removed_tensor = torch.cat((encoder_input_removed_tensor, encoder_input_removed[ueberlappen_len:]), 0)

            i += predict_len



    prediction_encoder = prediction_encoder.detach().numpy()
    prediction_encoder_sliding = sliding_window(prediction_encoder, windowSize) 
    #apply two-side sliding window
    # prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, 4000)   #apply two-side sliding window again to smooth more
    # print("in Sliding")
    for i in range(windowIterations):
        prediction_encoder_sliding = sliding_window(prediction_encoder_sliding, windowSize)

    return prediction_encoder,prediction_encoder_sliding, encoder_input_removed_tensor

#needed
def sliding_window(spline_array, window_size):
    length_spline_array = len(spline_array)
    result = np.array([])
    for index, number in enumerate(spline_array, start=0):
        if index == 0:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index == length_spline_array-1:
            result = np.concatenate((result,[spline_array[index]]), axis = 0)
        elif index < window_size:#index < 200
            window_size_copy = index
            spline_array_lower_window = spline_array[:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size_copy]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        elif index+window_size > length_spline_array-1:
            window_size_copy = length_spline_array - 1 - index 
            spline_array_lower_window = spline_array[index-window_size_copy:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:length_spline_array]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
        else:
            spline_array_lower_window = spline_array[index-window_size:index]
            middle_value = spline_array[index]
            spline_array_upper_window = spline_array[index+1:index+1+window_size]
            arr = np.concatenate((spline_array_lower_window,[middle_value],spline_array_upper_window), axis = 0)
            length = len(arr)
            sum = np.sum(arr)
            new_value = sum/length
            result = np.concatenate((result,[new_value]), axis = 0)
    
    return result

#needed
def plot_val_encoder_roundedInput(file_number: int, column: int):
    config = get_config() #config.py
    i2v = index_to_value_dict(config["vocab_size"], config["extra_tokens"])

    df = pd.read_csv(f"results_val/val_epoch_{file_number}.csv")
    x_values = np.arange(0,1000)[:]

    div_term = df[f"div_term_{column}"][0]
    min_value = df[f"min_value_{column}"][0]
    noise_std = df[f"noise_std_{column}"][0]

    groundTruth = torch.tensor(df[f"groundTruth_{column}"].to_numpy())
    noise = torch.tensor(df[f"noise_{column}"].to_numpy())
    noise_removed = torch.tensor(df[f"noise_removed_{column}"].to_numpy())
    prediction = torch.tensor(df[f"prediction_{column}"].to_numpy())

    noise = noise.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])
    noise_removed = noise_removed.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])
    
    #map indices to discrete values
    prediction = prediction.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])

    #denormalize discrete values
    noise = (noise*div_term)+min_value
    noise_removed = (noise_removed*div_term)+min_value
    prediction = (prediction*div_term)+min_value
    
    
    result = []
    result.append([[noise_removed, "Noise removed"], [groundTruth,"GroundTruth"], [prediction, "Prediction"]])

    length = len(result)
    _, axis = plt.subplots(length,1, sharex=True)
    axis.plot(x_values, noise, label=f"Noise", linewidth=4)
    axis.plot(x_values, noise_removed, label=f"Noise Interpolation", linewidth=4)
    axis.plot(x_values, prediction, label=f"Prediction", linewidth=4)

    plt.show()

#needed
def plot_train_encoder_roundedInput(file_number: int):
    config = get_config() #config.py
    i2v = index_to_value_dict(config["vocab_size"], config["extra_tokens"])

    df = pd.read_csv(f"results_train/train_epoch_{file_number}.csv")
    x_values = np.arange(0,1000)[:]

    div_term = df["div_term"][0]
    min_value = df["min_value"][0]
    noise_std = df["noise_std"][0]

    groundTruth = torch.tensor(df[f"groundTruth"].to_numpy())
    noise = torch.tensor(df[f"noise"].to_numpy())
    noise_removed = torch.tensor(df[f"noise_removed"].to_numpy())
    prediction = torch.tensor(df[f"prediction"].to_numpy())

    #map indices to discrete values
    noise = noise.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])
    noise_removed = noise_removed.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])
    prediction = prediction.type(torch.float32).apply_(lambda x: 0 if int(x) > config["vocab_size"] else i2v[f"{int(x)}"])

    #denormalize discrete values
    noise = (noise*div_term)+min_value
    noise_removed = (noise_removed*div_term)+min_value
    prediction = (prediction*div_term)+min_value

    result = []
    result.append([[noise_removed, "Noise removed"], [groundTruth,"GroundTruth"], [prediction, "Prediction"]])

    length = len(result)
    _, axis = plt.subplots(length,1, sharex=True)
    axis.plot(x_values, noise, label=f"Noise", linewidth=4)
    axis.plot(x_values, noise_removed, label=f"Noise Interpolation", linewidth=4)
    axis.plot(x_values, prediction, label=f"Prediction", linewidth=4)

    plt.show()

#needed
def predict_encoder_interpolation_roundedInput_CE_floatCurrent(seq_length: int, model_filename, y_noisy_spline,min_value_spline, max_value_spline, mask_indices,config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("Using device:", device)
    vocab_size = config["vocab_size"]
    vocab_extra_tokens = config["extra_tokens"]

    v2i = value_to_index_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    i2v = index_to_value_dict(vocab_size_numbers=vocab_size, vocab_extra_tokens=vocab_extra_tokens)
    exp = calc_exp(smallest_number=1/vocab_size)

    #build model
    model = build_encoder_interpolation_uknToken_projection(len(v2i), seq_length)
    # print(f'Preloading model {model_filename}')
    print("\tLoading Model")
    state = torch.load(model_filename, map_location=torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'])
    model.eval()    


    with torch.no_grad():
        #min-max scaling
        div_term = max_value_spline-min_value_spline
        encoder_input_removed = torch.tensor((y_noisy_spline-min_value_spline)/div_term,dtype=torch.float32)[:].to(device)

        #round normalized values to discrete values
        encoder_input_removed = round_numbers_individually(vocab_size,encoder_input_removed)[:]

        #map discrete values to indices and map masked values to index equivalent of "ukn"-token
        encoder_input_removed[mask_indices] = v2i["ukn"]
        encoder_input_removed = encoder_input_removed.apply_(lambda x: x if x>vocab_size else v2i[f"{round_with_exp(x, exp)}"]).type(torch.long)

        #prediction
        model_out = greedy_decode_timeSeries_paper_projection(model, encoder_input_removed.unsqueeze(0))


    #map indices to discrete values
    model_out = model_out.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])
    encoder_input_removed = encoder_input_removed.type(torch.float32).apply_(lambda x: 0 if int(x)>vocab_size else i2v[f"{int(x)}"])

    #denormalize discrete values
    model_out = (model_out*div_term)+min_value_spline
    encoder_input_removed = (encoder_input_removed*div_term)+min_value_spline


    return model_out, encoder_input_removed

#needed
def plot_multiple_pred_with_names_error_bar_area(x_values, y_values_prediction_total, title, labels, folderName, intervalle):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    # DIN A4 Breite in Zoll
    a4_width_inch = 6  # für volle Breite
    a4_height_inch = 11  # für volle Breite

    # DPI des Bildes (300 für Veröffentlichungen)
    dpi = 300

    # Schriftgröße in LaTeX-Punkten (z. B. 10pt, 12pt)
    latex_font_size_pt = 15

    # Umrechnung von Punkt in Zoll (1 pt = 1/72 Zoll)
    points_to_inches = 1 / 72

    # Umrechnung der Schriftgröße in Zoll
    latex_font_size_inch = latex_font_size_pt * points_to_inches



    labelSize = latex_font_size_pt
    labelSizeX = latex_font_size_pt -3
    labelSizeLegend = latex_font_size_pt
    lineWidth = 2
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(length,1, sharex=True, figsize=(a4_height_inch, a4_width_inch))
    for i in range(length):
        axis[i].tick_params(axis='both', which='both', labelsize=labelSize)
        # axis[i].set_title(title[i],fontsize=labelSize,fontweight='bold')
        axis[i].set_ylabel(labels[i],fontsize=labelSize)
        axis[i].margins(x=0.0)
    lines = []
    min_value = 0.0
    max_value = 0.0


    for i in range(length):
        specific_plot_data = y_values_prediction_total[i]
        num_plots = len(specific_plot_data)
        if i == 1:
            count = 0
            for j in range(num_plots):
                if j > len(specific_plot_data)-4:
                    count += 1
                    lower_std = specific_plot_data[j][0][0]
                    name_lower_std = specific_plot_data[j][0][1]
                    color_lower_std = specific_plot_data[j][0][2]

                    upper_std = specific_plot_data[j][1][0]
                    # name_upper_std = specific_plot_data[j][1][1]
                    # color_upper_std = specific_plot_data[j][1][2]

                    # axis[i].plot(x_values, lower_std, label=f"{name_lower_std}",linewidth=lineWidth, color=color_lower_std, alpha=0.35, linestyle='--')
                    # axis[i].plot(x_values, upper_std, label=f"{name_upper_std}",linewidth=lineWidth, color=color_upper_std, alpha=0.35, linestyle='--')
                    axis[i].fill_between(x_values, lower_std, upper_std, color=color_lower_std, alpha=0.3)
                    axis[2].fill_between(x_values, count, -count, color=color_lower_std, alpha=0.3)
                    
                else:
                    entry = specific_plot_data[j]
                    y_values = entry[0]
                    name = entry[1]
                    color = entry[2]
                    axis[i].plot(x_values, y_values, label=f"{name}",linewidth=lineWidth, color=color) 
        else:
            for j in range(num_plots):
                entry = specific_plot_data[j]
                y_values = entry[0]
                name = entry[1]
                color = entry[2]
                line, = axis[i].plot(x_values, y_values, label=f"{name}",linewidth=lineWidth, color=color)
                if i==2:
                    min_value_plot = min(y_values)
                    max_value_plot = max(y_values)

                    min_value = min_value_plot if min_value_plot < min_value else min_value
                    max_value = max_value_plot if max_value_plot > max_value else max_value
                if i == 0:
                    lines.append(line)



    # axis[0].set_ylabel("dV/dQ",fontsize=labelSize)
    # axis[1].set_ylabel("dV/dQ",fontsize=labelSize)
    # axis[2].set_ylabel("sigma",fontsize=labelSize)
    # axis[0].legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    axis[0].legend(loc='best',prop={'size': labelSizeLegend, 'weight': 'bold'})
    offset = (max_value-min_value)*0.05
    axis[2].set_ylim(min_value-offset, max_value+offset)
    # axis[2].set_ylim(-3, 3)

    plt.xlabel("Time Steps",fontsize=labelSizeX) #englisch
    # plt.xlabel("Zeitschritte",fontsize=labelSize) #deutsch

    # rax = plt.axes([0.4, 0.05, width, height])
    # lines_by_label = {l.get_label(): l for l in lines}
    # line_colors = [l.get_color() for l in lines]
    # check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props={'color': line_colors},
    # frame_props={'edgecolor': line_colors},
    # check_props={'facecolor': line_colors},)
    # for label in check.labels:
    #     label.set_fontsize(labelSize)
    # def callback(label):
    #         ln = lines_by_label[label]
    #         ln.set_visible(not ln.get_visible())
    #         ln.figure.canvas.draw_idle()
    # check.on_clicked(callback)
    # draggable_check = DraggableCheckButtons(check, axis[0], width, height)
    # draggable_check.connect()

    fmt = "svg"
    save_path = Path(f"save_figure/{folderName}/")
    save_path.mkdir(parents=True, exist_ok=True)
    file_path = save_path / f"{title}.{fmt}" if save_path else f"{title}.{fmt}"
    plt.gcf().savefig(file_path, format=fmt, dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    plot_val_encoder_roundedInput(320, 0)
    plot_val_encoder_roundedInput(320, 1)
    plot_val_encoder_roundedInput(320, 2)
    plot_val_encoder_roundedInput(320, 3)

    # plot_multiple_pred_with_names_error_bar_area(1430)




def plot_multiple_pred_with_names_multiple_error_bar(x_values, y_values_prediction_total, noise_std, title):
    '''
    y_values_prediction_total is an array containing "Noise_removed","GroundTruth","Prediction"
    important to calculate the error:
        - use "GroundTruth" name for reference timeseries
        - use "Prediction" for prediction timeseries
    '''
    labelSize = 18
    lineWidth = 3
    width = 0.15
    height = 0.2
    length = len(y_values_prediction_total)
    _, axis = plt.subplots(2,1, sharex=True)
    # fig.subtitle(title,fontsize=labelSize+2,fontweight='bold')
    for i in range(2):
        axis[i].tick_params(axis='both', which='both', labelsize=labelSize)
    lines = []
    linesBottom = []
    groundTruth = None
    prediction = None
    count = 0
    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        length = len(y_values_copy)
        y = 0
        while y < length:
            entry = y_values_copy[y]
            if (len(entry) != 2):
                y_values = entry[0]
                errorPlot = entry[1]
                name = entry[2]
                nameErrorPlot = entry[3]
                if(len(y_values) != 0 and len(errorPlot) != 0):
                    line1, = axis[0].plot(x_values, y_values, label=f"{name}",linewidth=lineWidth)
                    color1 = line1.get_color()
                    line2, = axis[1].plot(x_values, errorPlot, label=f"{nameErrorPlot}",linewidth=lineWidth,color=color1)
                    lines.append(line1)
                    linesBottom.append(line2)
            else:
                y_values = entry[0]
                name = entry[1]
                if(len(y_values) != 0):
                    line1, = axis[0].plot(x_values, y_values, label=f"{name}",linewidth=lineWidth)
                    lines.append(line1)

            y += 1


    rax = plt.axes([0.4, 0.05, width, height])
    rax_bottom = plt.axes([0.4, 0.05, width, height])

    lines_by_label = {l.get_label(): l for l in lines}
    line_colors = [l.get_color() for l in lines]

    lines_by_label_bottom = {l.get_label(): l for l in linesBottom}
    line_colors_bottom = [l.get_color() for l in linesBottom]

    check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props=
    {'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},)

    check_bottom = CheckButtons(ax=rax_bottom, labels=lines_by_label_bottom.keys(),actives=[l.get_visible() for l in lines_by_label_bottom.values()],label_props=
    {'color': line_colors_bottom},
    frame_props={'edgecolor': line_colors_bottom},
    check_props={'facecolor': line_colors_bottom},)
    
    for label in check.labels:
        label.set_fontsize(labelSize)

    for label in check_bottom.labels:
        label.set_fontsize(labelSize)

    def callback(label):
            ln = lines_by_label[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()

    def callback_bottom(label):
            ln = lines_by_label_bottom[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()
    
    check.on_clicked(callback)
    check_bottom.on_clicked(callback_bottom)

    draggable_check = DraggableCheckButtons(check, axis[0], width, height)
    draggable_check_bottom = DraggableCheckButtons(check_bottom, axis[1], width, height)
    
    draggable_check.connect()
    draggable_check_bottom.connect()

    axis[0].set_title("Predictions to Noisy Data",fontsize=labelSize,fontweight='bold')
    axis[1].set_title("Error rate with respect to noise standard deviation",fontsize=labelSize,fontweight='bold')
    plt.show()

def plot_multiple_pred_with_names(x_values, y_values_prediction_total,width, title=None, xLabel=None, yLabel=None):
    '''
    y_values_prediction_total is an array containing multiple timeseries
    '''
    labelSize = 18
    lineWidth = 3
    height = 0.2

    length = len(y_values_prediction_total)
    _, axis = plt.subplots(1,1, sharex=True)

    axis.tick_params(axis='both', which='both', labelsize=labelSize)
    if type(title) == str:
        axis.set_title(title,fontsize=labelSize,fontweight='bold')
    if type(xLabel)== str:
        plt.xlabel(xLabel,fontsize=labelSize)
    if type(yLabel) == str:
        plt.ylabel(yLabel,fontsize=labelSize)

    lines = []

    for i in range(length):
        y_values_copy = y_values_prediction_total[i]
        length = len(y_values_copy)
        y = 0
        while y < length:
            entry = y_values_copy[y]
            y_values = entry[0]
            name = entry[1]
            if(len(y_values) != 0):
                line, = axis.plot(x_values, y_values, label=f"{name}",linewidth=lineWidth)
                lines.append(line)
            y += 1

    rax = plt.axes([0.4, 0.05, width, height])
    lines_by_label = {l.get_label(): l for l in lines}
    line_colors = [l.get_color() for l in lines]
    check = CheckButtons(ax=rax, labels=lines_by_label.keys(),actives=[l.get_visible() for l in lines_by_label.values()],label_props={'color': line_colors},
    frame_props={'edgecolor': line_colors},
    check_props={'facecolor': line_colors},)
    for label in check.labels:
        label.set_fontsize(labelSize)
    def callback(label):
            ln = lines_by_label[label]
            ln.set_visible(not ln.get_visible())
            ln.figure.canvas.draw_idle()
    check.on_clicked(callback)
    draggable_check = DraggableCheckButtons(check, axis, width, height)
    draggable_check.connect()
    plt.show()
