import numpy as np
from PIL import Image
import os
from skimage.filters import threshold_otsu
import os
import pandas as pd
import matplotlib.pyplot as plt


def get_tif_files(folder_path):
    tif_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            tif_files.append(os.path.join(folder_path, file))
    return tif_files


def open_image(file_path):
    image = Image.open(file_path).convert('L')  # Convert image to grayscale
    return image


def binarize_image(image, threshold):
    binary_image = np.where(image > threshold, 1, 0)
    return binary_image


def count_white_pixels(image):
    white_pixels = np.sum(image)
    return white_pixels


def count_black_pixels(image):
    black_pixels = np.sum(1 - image)
    return black_pixels


def collect_data(folder_path, pixel_size):
    # Create an empty DataFrame to store the information
    table_data = pd.DataFrame(columns=['animal', 'brain_area', 'white_pixels', 'black_pixels', 'total_pixels'])
    
    # Create a folder for the control plots if it doesn't exist in the folder path
    if not os.path.exists(os.path.join(folder_path, 'control_plots')):
        os.mkdir(os.path.join(folder_path, 'control_plots'))

    # Create a list with all the tif files in the folder    
    file_list = get_tif_files(folder_path)

    # Loop through all the images in the folder
    for filepath in file_list:

        img_name = os.path.basename(filepath)
        animal = img_name.split('_')[0]
        brain_area = img_name.split('_')[1]

        # Open the image and calculate the threshold
        img = open_image(filepath)
        thr = threshold_otsu(np.array(img))
        if thr <= 10:
            thr = np.int32(20)
        img_bin = binarize_image(img, thr)

            
        # Calculate the area of the image and the area of the white and black pixels
        white_pixels = count_white_pixels(img_bin)
        black_pixels = count_black_pixels(img_bin)
        all_pixels = white_pixels + black_pixels

        area_image = all_pixels * pixel_size**2
        area_white = white_pixels * pixel_size**2
        area_black = black_pixels * pixel_size**2

        # Append the information to the DataFrame
        _temp_ = {'animal': animal, 'brain_area': brain_area, 'white_pixels': white_pixels, 
                  'black_pixels': black_pixels, 'total_pixels': all_pixels}
        table_data.loc[len(table_data)] = _temp_

        # Create a control plot for each image
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        area_image_mm = area_image / 1000  # Convert from um^2 to mm^2
        fig.suptitle(f'Animal {animal} | {brain_area} area: {area_image_mm:.2f} mm^2 | Threshold: {thr}', weight='bold')
        
        # Plot original image and binarized image
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Original image in grayscale')
        ax[0].set_xlabel('mm')
        ax[0].set_ylabel('mm')
        ax[1].imshow(img_bin, cmap='gray')
        ax[1].set_title('Binarized image')
        ax[1].set_xlabel('mm')
        ax[1].set_ylabel('mm')
        
        # Plot pie chart
        labels = ['White Pixels', 'Black Pixels']
        sizes = [area_white, area_black]
        colors = ['white', 'grey']
        ax[2].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
        ax[2].set_title('Pixel Distribution')
        ax[2].set_aspect('equal')  # Set aspect ratio to equal for circular pie chart

        # Transform x and y axes in ax[0] from pixels to mm
        x_ax_pixels = np.arange(0, img.width, 100)
        x_ax_mm = np.round(x_ax_pixels * pixel_size, decimals=1)
        y_ax_pixels = np.arange(0, img.height, 100)
        y_ax_mm = np.round(y_ax_pixels * pixel_size, decimals=1)

        for axnum in [0, 1]:            
            ax[axnum].set_xticks(x_ax_pixels)
            ax[axnum].set_yticks(y_ax_pixels)
            ax[axnum].set_xticklabels(x_ax_mm, rotation=45)
            ax[axnum].set_yticklabels(y_ax_mm)
            ax[axnum].set_xlabel('mm')
            ax[axnum].set_ylabel('mm')

        plt.tight_layout()

        # Save the figure
        figure_name = f'{animal}_{brain_area}_control_plot.png'
        figure_path = os.path.join(folder_path, 'control_plots', figure_name)
        plt.savefig(figure_path, dpi=300)        
        plt.close()

    # Compute the percentage of white pixels
    table_data['white_pixels_percentage'] = table_data['white_pixels'] / table_data['total_pixels'] * 100

    # Save the DataFrame as a csv file
    table_data.to_csv(os.path.join(folder_path, 'projections_quantification.csv'), index=False)

    return table_data


def plot_data(folder_path, datatable):
    
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
        ax[0].set_ylabel('% White Pixels')
        ax[0].set_xticks(np.arange(0, len(brain_areas)))
        ax[0].set_xticklabels(brain_areas, rotation=45)

    # Print the number of animals per brain area
    df = pd.DataFrame.from_dict(dict_animals_brain_area, orient='index')
    df['animals'] = df.apply(lambda row: ', '.join(row.astype(str)), axis=1)
    df['animals'] = df['animals'].apply(lambda x: x.replace(', None', ''))  # Remove None values
    df = df.iloc[:, 4:]  # Remove columns 0 to 3

    # Replace the index with the values in the new column
    df['brain_area_with_animals'] = df.index + ' (N=' + df['animals'].apply(lambda x: str(x.count(',') + 1)) + ')'
    df = df.set_index(df['brain_area_with_animals'])
    df = df.drop(columns=['brain_area_with_animals'])

    # Remove column and index names
    df.columns = [''] * len(df.columns)
    df.index.name = ''

    ax[1].clear()
    ax[1].set_title('Animals per Brain Area')
    ax[1].axis('off')
    ax[1].text(0.05, 0.1, df.to_string(header=False), fontsize=9)

    plt.tight_layout()

    # Save the figure
    figure_name = 'projections_quantification.png'
    figure_path = os.path.join(folder_path, figure_name)
    plt.savefig(figure_path, dpi=300)
    plt.close()



if __name__ == '__main__':
        
    pixel_size = 0.75521  # um, based on 20x objective and the printed table next to the HALO PC
    folderpath = r"S:\Shared\lab_members\sandovalortega_raqueladaia\Keyence_imaging\ACC_project\analysis\projections_quantification"

    # Collect the data and save it as a csv file
    quant_projections = collect_data(folderpath, pixel_size)

    # Plot the data
    plot_data(quant_projections)
