# Import libraries
import os
import numpy as np
import pandas as pd
from PIL import Image
# Import the function to compute the dynamic threshold
from skimage.filters import threshold_otsu
# Import plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec


def intensity_along_axis(img, ax=None):
    """ Compute the intensity along the x or y axis of the image. """
    if ax == 'x':
        s = np.nansum(img, axis=0)  # Sum along the x axis
    elif ax == 'y':
        s = np.nansum(img, axis=1) # Sum along the y axis
    elif ax is None:
        s = (np.nansum(img, axis=0), np.nansum(img, axis=1))  # Sum along the x and y axis
    return s

def remove_spines_plot(ax, loc=['all']):
    """Remove the spines of the plot."""
    if len(loc) == 1 and loc[0] == 'all':
        loc = ['top', 'right', 'bottom', 'left']
        [ax.spines[l].set_visible(False) for l in loc]
    else:
        [ax.spines[l].set_visible(False) for l in loc]

def remove_ticks_plot(ax, loc='all'):
    """Remove the ticks of the plot."""
    if loc == 'all':
        ax.set_xticks([])
        ax.set_yticks([])
    elif loc == 'x':
        ax.set_xticks([])
    elif loc == 'y':
        ax.set_yticks([])

def plot_intensity_along_axis(ax, x, y, loc):
    if loc == 'y':
        ax.fill_betweenx(x, y, color='grey', alpha=0.5)
    elif loc == 'x':
        ax.fill_between(x, y, color='grey', alpha=0.5)
    ax.margins(0)

def control_plot_steps(ax, img, title, axs_info):
    """
    Create the subplots of the control plots for each input image.

    Parameters:
    - ax (list of matplotlib Axes): List of Axes objects to plot the images.
    - img (numpy array): Input image data.
    - title (str): Title for the plot.
    - axs_info (dict or list): Information about the axes.

    Returns:
    - ax (list of matplotlib Axes): List of Axes objects with the plotted images.
    """

    # Compute x and y axis signals
    x_ax_signal, y_ax_signal = intensity_along_axis(img)

    # Define the label of the quantification plots
    if title != 'Mask':
        label_quant = 'Signal\nintensity'
    else:
        label_quant = 'Dimensions\nmask'

    # Plot original image, binarized image and area of the original image
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(title, pad=12)
    # Beautify
    remove_ticks_plot(ax[0], loc='x')
    remove_spines_plot(ax[0], loc=['top', 'right', 'bottom'])
       
    # Plot the signal in the y axis
    plot_intensity_along_axis(ax[1], np.arange(len(y_ax_signal)), y_ax_signal[::-1], 'y')
    # Set x label at given location
    ax[1].set_xlabel(label_quant, loc='center', rotation=45, color='grey')
    ax[1].xaxis.set_label_coords(1, -0.01)
    # Beautify
    remove_ticks_plot(ax[1])
    remove_spines_plot(ax[1])

    # Plot the signal in the x axis
    plot_intensity_along_axis(ax[2], np.arange(len(x_ax_signal)), -x_ax_signal, 'x')
    # Beautify
    remove_ticks_plot(ax[2], loc='y')
    remove_spines_plot(ax[2], loc=['top', 'right', 'left'])
    
    # Match the size of ax[1] and ax[2] to ax[0]    
    ax0_size = ax[0].get_position()
    ax1_size = ax[1].get_position()
    ax[1].set_position([ax0_size.x0 + ax0_size.width, ax0_size.y0, ax1_size.width, ax0_size.height])
    ax2_size = ax[2].get_position()
    ax[2].set_position([ax0_size.x0, ax0_size.y0 - ax2_size.height, ax0_size.width, ax2_size.height])

    # Set x ticks and labels for the medio-lateral axis
    x_ax_ticks = axs_info[0]
    x_ax_labels = axs_info[1]   
    ax[2].set_xticks(x_ax_ticks)
    ax[2].set_xticklabels(x_ax_labels, rotation=45)
    ax[2].set_xlabel('Medio-Lateral axis (\u03bcm)')

    # Set y ticks and labels for the dorso-ventral axis
    y_ax_ticks = axs_info[2]
    y_ax_labels = axs_info[3]
    ax[0].set_yticks(y_ax_ticks)
    ax[0].set_yticklabels(y_ax_labels)
    ax[0].set_ylabel('Dorso-Ventral axis (\u03bcm)')

    return ax
    

def get_tif_files(folder_path):
    """Get all the tif files in the folder provided."""
    tif_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            tif_files.append(os.path.join(folder_path, file))
    return tif_files


def collect_info_from_filename(filename):
    img_name = os.path.basename(filename)
    img_name = img_name.split('.')[0]
    animal = img_name.split('_')[0]
    brain_area = img_name.split('_')[1]
    return animal, brain_area

def open_tif_image(img_path):
    return Image.open(img_path)

def generate_mask(img, shape_rec):
    """Generate a mask for the image provided."""
    # For rectangular images
    if shape_rec:    
        mask = np.full_like(img, False)
        if len(np.array(img).shape) == 3:
            mask = mask[:, :, 0]
        else:
            mask = mask
    # For non-rectangular images
    if not shape_rec:
        image_matrix = np.array(img)
        if len(np.array(img).shape) == 3:
            mask = np.all(image_matrix == [0, 0, 0], axis=-1)
        elif len(np.array(img).shape) == 2:
            mask = np.all(image_matrix == [0, 0], axis=-1)
    return mask

def convert_image_to_gray(img):
    return img.convert('L')

def collect_within_mask(img, mask):
    return np.where(mask, np.nan, img)

def collect_image_mask(img_path, img_rectangle_shape):
    """Open the image and cut it to the area that contains tissue."""
    img = open_tif_image(img_path)
    # Find the area of the image that cotains tissue
    mask = generate_mask(img, img_rectangle_shape)
    # Convert image to grayscale
    image_gray = convert_image_to_gray(img)
    # Cut the image to the area that contains tissue
    cut_image = collect_within_mask(image_gray, mask)
    return cut_image, mask

def binarize_image(image, threshold):
    """Binarize the image using the threshold provided."""
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image

def count_pixels(image):
    """Count the number of white and black pixels in the image."""
    white_pixels = np.sum(image)
    black_pixels = np.sum(1 - image)
    all_pixels = white_pixels + black_pixels
    return [white_pixels, black_pixels, all_pixels]

def compute_area(img, pixel_size):
    """Compute the area of the image and the area of the white and black pixels."""
    counts = count_pixels(img)
    [w, b, all] = [c * pixel_size**2 for c in counts]
    return w, b, all

def compute_threshold(img):
            non_nan_values = img[~np.isnan(img)]
            thr = threshold_otsu(np.array(non_nan_values))
            # If the threshold is too low, set it to 20
            if thr <= 10:
                thr = np.int32(20)
            return thr

def make_figures_pdf_editable():
    """Make the figures editable in Adobe Illustrator."""
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

def prepare_axis_information(img, pixel_size):
    x_ax_pixels = np.linspace(0, img.shape[1], 6)
    x_ax_um = np.round(x_ax_pixels * pixel_size, decimals=1)
    y_ax_pixels = np.linspace(0, img.shape[0], 6)
    y_ax_um = np.round(y_ax_pixels * pixel_size, decimals=1)
    return [x_ax_pixels, x_ax_um, y_ax_pixels, y_ax_um]

def save_table(df, folderpath, filename):
    filepath = os.path.join(folderpath, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    df.to_csv(filepath, index=False)      


def generate_control_plot(img, img_bin, msk, pixel_size, info_pie):

    # Create a grid for the subplots
    gs = gridspec.GridSpec(13, 11)

    # Prepare axis informationTransform x and y axes from pixels to mm
    axis_info = prepare_axis_information(img, pixel_size)

    # Subplot original image
    ax_ori = [plt.subplot(gs[0:4, 0:4]), 
                plt.subplot(gs[0:4, 4]), 
                plt.subplot(gs[4, 0:4])]
    ax_ori = control_plot_steps(ax_ori, img, 'Grayscale image', axis_info)
    
    # Subplot binarized image
    ax_bin = [plt.subplot(gs[0:4, 6:10]), 
                plt.subplot(gs[0:4, 10]), 
                plt.subplot(gs[4, 6:10])]
    ax_bin = control_plot_steps(ax_bin, img_bin, 'Binarized image', axis_info)

    # Subplot area of the original image
    ax_msk = [plt.subplot(gs[7:11, 0:4]),
                plt.subplot(gs[7:11, 4]), 
                plt.subplot(gs[11, 0:4])]
    ax_msk = control_plot_steps(ax_msk, ~msk, 'Mask', axis_info)
            
    # Subplot pie chart
    ax_pie = plt.subplot(gs[7:11, 6:11])
    ax_pie.pie(info_pie["sizes"], labels=info_pie["labels"], autopct='%1.1f%%', startangle=90, colors=info_pie["colors"], 
                labeldistance=1.2,  wedgeprops = {'edgecolor': 'black', 'linewidth': 0.5, 'antialiased': True})
    # set title to subplot
    ax_pie.set_title('Proportion of signal vs background', pad=12)
    # Set aspect ratio to equal for a circular pie chart
    ax_pie.set_aspect('equal')
    # Add a line to the pie chart
    ax_pie.add_line(plt.Line2D([0.5, 0.5], [0.5, 0.5], color='black', linewidth=1))
    

def collect_data(folder_path, pixel_size, img_shape):
    """
    Collect the data from the images in the folder provided and save it as a csv file.
    Create a control plot for each image.
    """

    # Set fonts editable in Adobe Illustrator
    make_figures_pdf_editable()

    # Create an empty DataFrame to store the information
    table_data = pd.DataFrame()
    table_data_axis = pd.DataFrame()
    
    # Create a folder for the control plots if it doesn't exist in the folder path
    if not os.path.exists(os.path.join(folder_path, 'control_plots')):
        os.mkdir(os.path.join(folder_path, 'control_plots'))

    # Create a list with all the tif files in the folder    
    file_list = get_tif_files(folder_path)
    if len(file_list) == 0:
        raise ValueError("No tif files found in the folder path")
    n_images = len(file_list)

    # Loop through all the images in the folder
    for i, filepath in enumerate(file_list):

        # Print the progression every 5th iteration
        if int(i+1) % 5 == 0:
            print(f"Processing image {int(i+1)} out of {n_images}")

        # Get the animal and brain area from the image name
        animal, brain_area = collect_info_from_filename(filepath)
        
        # Open the image, collect the mask, calculate threshold for binarization and binarize the image
        img, msk = collect_image_mask(filepath, img_shape)
        thr = compute_threshold(img)
        img_bin = binarize_image(img, thr)

        # Calculate the number of white and black pixels of the pixels within the mask and the area they occupy
        [w, b, all] = count_pixels(img_bin[~msk])
        area_w, area_b, area_img = compute_area(img_bin[~msk], pixel_size)

        # Append the information to the DataFrame for the image
        _temp_ = {'animal': animal, 
                  'brain_area': brain_area, 
                  'pixels_signal': w, 
                  'pixels_black': b, 
                  'pixels_total': all,
                  'threshold': thr, 
                  'area_image': area_img,
                  'area_signal': area_w,
                  'area_black': area_b}
        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=_temp_.keys())
        table_data.loc[len(table_data)] = _temp_

        # Append the information to the DataFrame for the axis
        _temp_axis_ = {'animal': animal,
                       'brain_area': brain_area,
                       'signal_bin_x_ax': intensity_along_axis(img_bin, 'x'),
                       'signal_bin_y_ax': intensity_along_axis(img_bin, 'y'),
                       'signal_gray_x_ax': intensity_along_axis(img, 'x'),
                       'signal_gray_y_ax': intensity_along_axis(img, 'y')}
        if np.sum(table_data_axis.shape) == 0:
            table_data_axis = pd.DataFrame(columns=_temp_axis_.keys())
        table_data_axis.loc[len(table_data_axis)] = _temp_axis_
        
        del _temp_, _temp_axis_
        
        # Generate control plot
        fig = plt.figure(figsize=(8, 8))
        area_image_um = area_img / 1000
        fig.suptitle(f'Animal {animal} | {brain_area} | Area: {area_image_um:.2f}\u03bcm\u00b2 | Threshold: {thr:.2f}', weight='bold')
        info_pie = {"labels": ['Area receiving\nprojections', 'Area without\nprojections'],
                    "sizes": [area_w, area_b],
                    "colors": ['white', 'grey']}
        generate_control_plot(img, img_bin, msk, pixel_size, info_pie)
        
        # Save control plot
        figure_name = f'{animal}_{brain_area}_control_plot.pdf'
        figure_path = os.path.join(folder_path, 'control_plots', figure_name)
        plt.savefig(figure_path, dpi=300)        
        plt.close(fig)
        
    print("Processing images finished")


    # Compute the percentage of white pixels
    table_data['percent_signal'] = table_data['pixels_signal'] / table_data['pixels_total'] * 100

    # Save dataframes as csv files
    save_table(table_data, folderpath, 'projections_quantification.csv')
    save_table(table_data_axis, folderpath, 'projections_quantification_axis.csv')

    return table_data, table_data_axis


def plot_summary_data(folder_path, df):

    # Set fonts editable in Adobe Illustrator
    make_figures_pdf_editable()

    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    brain_areas = df['brain_area'].unique()
    fig.suptitle('Projections Quantification', weight='bold')

    # Set the color for the bars
    color_bars = 'royalblue'

    # Create a dictionary with the animals per brain area
    dict_animals_brain_area = {}
    
    # Plot the bar plot with the mean and sem of the white pixels percentage
    for i_ba, br_ar in enumerate(brain_areas):
        
        # Get the data for the brain area
        data = df[df['brain_area'] == br_ar]
        vals = data['percent_signal']
        vals_mean = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals))

        # Fill the dictionary with the animals in the brain area
        dict_animals_brain_area[br_ar] = data['animal'].unique().tolist()

        # Plot the bar plot with the mean and sem
        ax[0].bar(i_ba, vals_mean, color=color_bars, yerr=sem)

        # Plot the scatter plot with round markers
        ax[0].scatter(np.repeat(i_ba, len(vals)), vals, edgecolors=color_bars, facecolors='white')

    # Beautify
    yval_max = np.ceil(ax[0].get_ylim()[1] / 10) * 10
    ax[0].set_ylabel("% of brain area receiving projections")
    ax[0].set_yticks(np.linspace(0, yval_max, 6))
    ax[0].set_xticks(np.arange(0, len(brain_areas)))
    ax[0].set_xticklabels(brain_areas, rotation=45)
    ax[0].set_ylim(0, yval_max)

    # Print the number of animals per brain area
    df = pd.DataFrame.from_dict(dict_animals_brain_area, orient='index')
    df['animals'] = df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
    df['animals'] = df['animals'].apply(lambda x: x.replace(', None', ''))  # Remove None values
    i_anim_col = df.columns.get_loc("animals")
    df = df.iloc[:, i_anim_col:]  # Remove columns 0 to the one with all the animals

    # Replace the index with the values in the new column
    df['brain_area_with_animals'] = df.index + ' (N=' + df['animals'].apply(lambda x: str(x.count(',') + 1)) + ')'  # Count the number of animals
    df = df.set_index(df['brain_area_with_animals'])
    df = df.drop(columns=['brain_area_with_animals'])

    # Remove column and index names
    df.columns = [''] * len(df.columns)
    df.index.name = ''

    ax[1].set_title('Animals per Brain Area')
    ax[1].axis('off')
    ax[1].text(0.05, 0.1, df.to_string(header=False), fontsize=9)

    # Set the layout
    plt.tight_layout()
    sns.despine(fig=fig, ax=ax[0], top=True, right=True)

    # Delete the DataFrame
    del df

    # Save the figure
    figure_name = 'projections_quantification.pdf'
    figure_path = os.path.join(folder_path, figure_name)
    plt.savefig(figure_path, dpi=300)
    plt.close()



if __name__ == '__main__':
        
    pixel_size = 0.75521  # um, based on 20x objective and the printed table next to the HALO PC
    folderpath = r"S:\Shared\custom_data_analysis\volume_projections\test_dataset"
    # folderpath = r"S:\Shared\lab_members\sandovalortega_raqueladaia\Manuscripts\2024_axon_quantification\fig3_compre_fluorophores\quantification_GFP"
    is_image_rectangle = False

    # Collect the data and save it as a csv file
    quant_projections, _ = collect_data(folderpath, pixel_size, is_image_rectangle)

    # Plot the data
    plot_summary_data(folderpath, quant_projections)
