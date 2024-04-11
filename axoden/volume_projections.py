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


def plot_data_images(ax, img, title, axs_info):
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
    x_ax_signal = np.nansum(img, axis=0)
    y_ax_signal = np.nansum(img, axis=1)

    # Define the label of the quantification plots
    if title != 'Mask':
        label_quant = 'Signal\nintensity'
    else:
        label_quant = 'Dimensions\nmask'

    # Plot original image, binarized image and area of the original image
    ax[0].imshow(img, cmap='gray')
    ax[0].set_title(title, pad=12)
    ax[0].set_xticks([])
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    
    # Plot the signal in the y axis
    ax[1].fill_betweenx(np.arange(len(y_ax_signal)), y_ax_signal[::-1], color='grey', alpha=0.5)
    ax[1].set_xlabel(label_quant, loc='center', rotation=45, color='grey')
    ax[1].margins(0)
    # Change location of the xlabel
    ax[1].xaxis.set_label_coords(1, -0.01)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['left'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    # Plot the signal in the x axis
    ax[2].fill_between(np.arange(len(x_ax_signal)), -x_ax_signal, color='grey', alpha=0.5)
    ax[2].margins(0)
    ax[2].set_yticks([])
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['left'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    
    # Match the size of ax[1] and ax[2] to ax[0]    
    ax0_size = ax[0].get_position()
    ax1_size = ax[1].get_position()
    ax[1].set_position([ax0_size.x0 + ax0_size.width, ax0_size.y0, ax1_size.width, ax0_size.height])
    ax2_size = ax[2].get_position()
    ax[2].set_position([ax0_size.x0, ax0_size.y0 - ax2_size.height, ax0_size.width, ax2_size.height])

    # Set x ticks and labels 
    x_ax_ticks = axs_info[0]
    x_ax_labels = axs_info[1]   
    ax[2].set_xticks(x_ax_ticks)
    ax[2].set_xticklabels(x_ax_labels, rotation=45)
    ax[2].set_xlabel('Medio-Lateral axis (\u03bcm)')

    # Set y ticks and labels 
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


def open_image(file_path, img_rectangle_shape):
    """Open the image and cut it to the area that contains tissue."""
    image = Image.open(file_path)
    mask = np.full_like(image, False)
    if len(np.array(image).shape) == 3:
        mask = mask[:, :, 0]
    else:
        mask = mask
    # Find the area of the image that cotains tissue
    if not img_rectangle_shape:
        image_matrix = np.array(image)
        if len(np.array(image).shape) == 3:
            mask = np.all(image_matrix == [0, 0, 0], axis=-1)
        elif len(np.array(image).shape) == 2:
            mask = np.all(image_matrix == [0, 0], axis=-1)
    # Convert image to grayscale
    image_gray = image.convert('L')
    # Cut the image to the area that contains tissue
    cut_image = np.where(mask, np.nan, image_gray)
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


def collect_data(folder_path, pixel_size, img_shape):
    """
    Collect the data from the images in the folder provided and save it as a csv file.
    Create a control plot for each image.
    """

    # Set fonts editable in Adobe Illustrator
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'

    # Create an empty DataFrame to store the information
    table_data = pd.DataFrame()
    
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
        img_name = os.path.basename(filepath)
        img_name = img_name.split('.')[0]
        animal = img_name.split('_')[0]
        brain_area = img_name.split('_')[1]

        # Open the image and calculate the threshold
        img, msk = open_image(filepath, img_shape)
        non_nan_values = img[~np.isnan(img)]
        thr = threshold_otsu(np.array(non_nan_values))
        # If the threshold is too low, set it to 20
        if thr <= 10:
            thr = np.int32(20)
        img_bin = binarize_image(img, thr)

        # Calculate the number of white and black pixels
        [w, b, all] = count_pixels(img_bin[~msk])

        # Calculate the area of the image and the area of the white and black pixels
        area_image = all * pixel_size**2
        area_white = w * pixel_size**2
        area_black = b * pixel_size**2

        # Append the information to the DataFrame
        _temp_ = {'animal': animal, 
                  'brain_area': brain_area, 
                  'pixels_signal': w, 
                  'pixels_black': b, 
                  'pixels_total': all, 
                  'signal_bin_x_ax': np.nansum(img_bin, axis=0),
                  'signal_bin_y_ax': np.nansum(img_bin, axis=1),
                  'signal_gray_x_ax': np.nansum(img, axis=0),
                  'signal_gray_y_ax': np.nansum(img, axis=1),
                  'threshold': thr, 
                  'area_image': area_image,
                  'area_signal': area_white,
                  'area_black': area_black}
        if np.sum(table_data.shape) == 0:
            table_data = pd.DataFrame(columns=_temp_.keys())
        table_data.loc[len(table_data)] = _temp_
        del _temp_

        # Create a control plot for each image
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(13, 11)

        # Prepare axisTransform x and y axes from pixels to mm
        x_ax_pixels = np.linspace(0, img.shape[1], 6)
        x_ax_um = np.round(x_ax_pixels * pixel_size, decimals=1)
        y_ax_pixels = np.linspace(0, img.shape[0], 6)
        y_ax_um = np.round(y_ax_pixels * pixel_size, decimals=1)
        
        area_image_um = area_image / 1000  # Convert from mm^2 to um^2
        fig.suptitle(f'Animal {animal} | {brain_area} | Area: {area_image_um:.2f}\u03bcm\u00b2 | Threshold: {thr:.2f}', weight='bold')

        # Subplot original image
        ax_ori = [plt.subplot(gs[0:4, 0:4]), 
                  plt.subplot(gs[0:4, 4]), 
                  plt.subplot(gs[4, 0:4])]
        ax_ori = plot_data_images(ax_ori, img, 'Original image in grayscale', [x_ax_pixels, x_ax_um, y_ax_pixels, y_ax_um])
        
        # Subplot binarized image
        ax_bin = [plt.subplot(gs[0:4, 6:10]), 
                  plt.subplot(gs[0:4, 10]), 
                  plt.subplot(gs[4, 6:10])]
        ax_ori = plot_data_images(ax_bin, img_bin, 'Binarized image', [x_ax_pixels, x_ax_um, y_ax_pixels, y_ax_um])

        # Subplot area of the original image
        ax_msk = [plt.subplot(gs[7:11, 0:4]),
                  plt.subplot(gs[7:11, 4]), 
                  plt.subplot(gs[11, 0:4])]
        ax_msk = plot_data_images(ax_msk, ~msk, 'Mask', [x_ax_pixels, x_ax_um, y_ax_pixels, y_ax_um])
                
        # Subplot pie chart
        ax_pie = plt.subplot(gs[7:11, 6:11])
        ax_pie.set_title('Area with and without projections', pad=12)
        labels = ['Area receiving\nprojections', 'Area without\nprojections']
        sizes = [area_white, area_black]
        colors = ['white', 'grey']
        ax_pie.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, labeldistance=1.2,  
                   wedgeprops = {'edgecolor': 'black', 'linewidth': 0.5, 'antialiased': True})
        ax_pie.set_aspect('equal')  # Set aspect ratio to equal for circular pie chart
        # Add a line to the pie chart
        ax_pie.add_line(plt.Line2D([0.5, 0.5], [0.5, 0.5], color='black', linewidth=1))
        
        # Save the figure
        figure_name = f'{animal}_{brain_area}_control_plot.pdf'
        figure_path = os.path.join(folder_path, 'control_plots', figure_name)
        plt.savefig(figure_path, dpi=300)        
        plt.close(fig)
        
    print("Processing images finished")

    # Compute the percentage of white pixels
    table_data['percent_signal'] = table_data['pixels_signal'] / table_data['pixels_total'] * 100

    # Save the DataFrame as a csv file
    table_filepath = os.path.join(folder_path, 'projections_quantification.csv')
    if os.path.exists(table_filepath):
        os.remove(table_filepath)
    table_data.to_csv(table_filepath, index=False)

    return table_data


def plot_data(folder_path, datatable):

    
    # Set fonts editable in Adobe Illustrator
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = 'Arial'
    
    # Create a figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    brain_areas = datatable['brain_area'].unique()
    fig.suptitle('Projections Quantification', weight='bold')

    # Set the color for the bars
    color_bars = 'royalblue'

    # Create a dictionary with the animals per brain area
    dict_animals_brain_area = {}
    
    # Plot the bar plot with the mean and sem of the white pixels percentage
    for i_brar, br_ar in enumerate(brain_areas):
        
        # Get the data for the brain area
        data = datatable[datatable['brain_area'] == br_ar]
        brar_animals = data['animal'].unique()
        white_pixels = data['white_pixels_percentage']
        mean_white_pixels_percent = np.mean(data['white_pixels_percentage'])
        sem_white_pixels_percent = np.std(data['white_pixels_percentage']) / np.sqrt(len(white_pixels))

        # Fill the dictionary with the animals in the brain area
        dict_animals_brain_area[br_ar] = brar_animals.tolist()

        # Plot the bar plot with the mean and sem
        ax[0].bar(i_brar, mean_white_pixels_percent, color=color_bars, yerr=sem_white_pixels_percent)

        # Plot the scatter plot with round markers
        ax[0].scatter(np.repeat(i_brar, len(white_pixels)), white_pixels, edgecolors=color_bars, facecolors='white')
        ax[0].set_ylim(0, 100)
        ax[0].set_ylabel("% of brain area receiving projections")
        ax[0].set_xticks(np.arange(0, len(brain_areas)))
        ax[0].set_xticklabels(brain_areas, rotation=45)

    # Print the number of animals per brain area
    df = pd.DataFrame.from_dict(dict_animals_brain_area, orient='index')
    df['animals'] = df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
    df['animals'] = df['animals'].apply(lambda x: x.replace(', None', ''))  # Remove None values
    i_anim_col = df.columns.get_loc("animals")
    df = df.iloc[:, i_anim_col:]  # Remove columns 0 to the one with all the animals

    # Replace the index with the values in the new column
    df['brain_area_with_animals'] = df.index + ' (N=' + df['animals'].apply(lambda x: str(x.count(',') + 1)) + ')'
    df['animals'] = df['animals'].apply(lambda x: x.replace('.0', ''))  # Remove None values
    df = df.set_index(df['brain_area_with_animals'])
    df = df.drop(columns=['brain_area_with_animals'])

    # Remove column and index names
    df.columns = [''] * len(df.columns)
    df.index.name = ''

    ax[1].clear()
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
    # folderpath = r"S:\Shared\custom_data_analysis\volume_projections\test_dataset"
    folderpath = r"S:\Shared\lab_members\sandovalortega_raqueladaia\Manuscripts\2024_axon_quantification\fig3_compre_fluorophores\quantification_GFP"
    is_image_rectangle = False

    # Collect the data and save it as a csv file
    quant_projections = collect_data(folderpath, pixel_size, is_image_rectangle)

    # Plot the data
    # plot_data(folderpath, quant_projections)
