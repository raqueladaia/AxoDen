import streamlit as st


def how_to_app():
    st.title('How To Use AxoDen ')

    st.markdown('''#### Disclaimer''')
    st.markdown('''
        Data uploaded to the web app is kept in memory for the duration of the analysis.  
        The memory is erased when closing or refreshing the browser.  
        No third party can access the data in memory.  
        The developers of AxoDen have no access to any uploaded nor generated data on the AxoDen web app.  
        Only the user can see, interact and store the data the user has uploaded and generated.  
        ''')

    st.markdown('''## Part 1: Prepare your images''')

    st.markdown('''
        Preparing your images for AxoDen requires the use of 3rd party software.  
                  
        We recommend using ImageJ or FIJI to modify your images because programs such as Adobe Illustrator  
        can redistribute the color information into 3 or more channels when saving the modified image as .tif.  
           
        AxoDen is unable to process more than 1 channel containing pixel values above 0.  
          
        - Download ImageJ or FIJI: [imagej.net](https://imagej.net/)    
        
                ''')
    
    st.markdown('''#### Steps''')

    st.markdown('''
                
        1.	Make sure the format of your images is .tif.  
                
        2.	Rename the files of your images with the following format:  
            *subjectID_brainregion_group_[additinal information if needed].tif.*  
            - Do not use spaces in the name of your file.  
            - AxoDen interprets the underscore sign as a spacer.  
            - AxoDen ignores any information contained in the file name that is beyond the 3rd underscore sign.  
            - If you have only one experimental group, you still need to fill the space for group  
            *i.e. "group", "control", "none"...*  
            - If you have ROIs of two names, use initials or separate them using a dash sign  
            *i.e. "Centrolateral thalamus" can be named as "CL", "TH-CL", "THCL", "centro-lateral-thalamus"...*  

        3.	If you want to mask your images to analyze a precise brain region:  
            3.1. Use Adobe Illustrator or Power Point to overlay the brain atlas to your image and set the boundaries of your brain region of interest.   
            3.2. Open the image of interest in ImageJ/FIJI.  
            3.3. Select the freehand selection tool on the top menu.        
            3.4. Draw the boundaries of the region of interest according to the overlay you created in step 3a.  
            3.5. On the top menu select in order:  
                3.5.1. Edit > Clear Outside.  
                3.5.2.	Image > Crop.  
            3.6.	If the signal to noise ratio needs to be enhanced:  
                3.6.1.	On the top menu select Image > Adjust > Brightness/Contrast  
                3.6.2.	Decrease the “Maximum” value using the sliding bar to increase fluorescence intensity.  
                3.6.3.	Increase the “Minimum” value using the sliding bar to decrease background fluorescence.  
                ***TIP:** Use the histogram (Analyze > Histogram) to confirm that the background fluorescence in the brain region does not contain zero values.*  
            3.7.	Save the resulting image following AxoDen naming convention.  
            
                ''')
    
    
    st.markdown('''## Part 2: AxoDen analysis''')

    st.markdown('''#### Steps''')

    st.markdown('''
                
        1.	Give a name to your project.  
                
        2.  Change the pixel size if you are using an objective other than the default 20x.  
            ***TIP:** If you are not interested on the ROIs' areas, you don't need to change this value*  
                
        3.  Toggle the check box if your images are masked.  
                
        2.	Drag and drop all the files you want to analyze.  
            
        3.	Right after the files have been uploaded, the analysis runs automatically.  
                
        4.	Inspect the control plots to make sure the masking of the image is accurate.  
            - The mask appears on the second row of the first column.  
            - The mask is shown in white.  
            - The mask should cover any part of the image that contains tissue.  
            - If the mask present black spots within, that is because the background fluorescence of the tissue is 0.  
            - To correct the mask, repeat Part 1, steps 3e to h to create a new masked image.  
            - If you cannot correct the mask, it means that during acquistion the background fluorescence was too low. **Consider retaking the image.**  
              
        5.	Inspect the control plots to make sure the signal does not bleed to surrounding areas.  
            - In this case, you want to examine the binarized image.  
            - The binarized image appears on the first row of the second column.  
            - The signal is shown in white.  
            - If the signal extends beyond the axon limits (=bleeding to surrounding areas),  
                it means the signal intensity is too high.  
            - To correct, repeat Part 1, steps 3e to h to create a new masked image.  
            - If you cannot correct the bleeding, your image is likely overexposed. **Consider retaking the image.**    
              

        6.	Download the control plots. 
            - Click the “Download figure as pdf” button.  
                
        7.  Inspect the Summary Data.  
            - The Summary Data figure is a collection of 3 plots.  
            - These are bar plots showing the data as mean +/- SEM.  
            - Each dot represents one animal/subject.  
            - The plot on the first column shows the *"% of innervation"* of all ROIs.  
                These are the final results of AxoDen that you can use for your publication.  
            - The plots on the second and third columns are "sanity checks".  
            - The plot of the second column shows the *"Area"* of each ROI.  
                You want low variance (= short error bars) for each ROI. Low variance means all selected ROIs have similar sizes.  
            - The plot of the third column shows the *"Threshold"* selected with the Otsu method to separate signal from background within the tissue.  
                Low variances (= short error bars) indicate that images have similar fluorescence intensity values.  

        8. Download the Summary Data figure.  
            - Click the “Download figure as pdf” button. 
          
        9. The Summary Data table is a table in tidy format that you can download to run your own statistics.  
            - "white_pixels" refer to the signal.  
            - "black_pixels" refer to the background.  
            - The download file is a .csv file.  
            - You can open the .csv file in Excel.  
            - You can copy-paste columns to Prism for further statistical analysis.  

        10. Download Summary Data table
            - Click on the arrow pointing downwards on the top right of the table.
        
        **If you are interested in the fluorescence intensity projected to the X and the Y axis, read the next steps:**  
        
        11. Inspect the Summary Data Axis.  
            - The grey lines represent individual animals/subjects.  
            - The blue line and the shade around it represent the mean +/- SEM, respectively.  
            - Different gray lines may have different sizes.  
            ***TIP:** Rotate cortical brain regions to align to the X or Y axis and investgate the innervation patterns of the different cortical layers.*  

        12. Download the Summary Data Axis figure.  
            - Click the “Download figure as pdf” button.  
        
        13. The Summary Data Axis table.
            - This table is not in tidy format.
            - The intensity values along the axes are stored as arrays in one cell.  
            - We recommend to use Python or MATLAB for the statistical analysis of this data.  

        10. Download Summary Data Axis table  
            - Click on the arrow pointing downwards on the top right of the table.         


        **All figures are created so that they can be modified in Adobe Illustrator.**  
                
    ''')

st.set_page_config(
    page_title="Axoden Tutorial & How To",
    page_icon="❔️"
)

how_to_app()
