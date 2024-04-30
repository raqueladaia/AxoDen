import streamlit as st

def how_to_app():
    st.title('How To Use AxoDen ')
    st.markdown('''
        ## Part 1: Prepare your images
        1.	Make sure the format of your images is .tif.
        2.	Rename the files of your images with the following format:  
                *subjectID_brainregion_[whatever you want].tif.*  
                
                ***TIP 1:** Do not use spaces in the name of your file.*  
                ***TIP 2:** AxoDen interprets the underscore sign (“_”) as a spacer. 
                If you want to differentiate subregions within one brain region (i.e. centrolateral thalamus vs. centromedial thalamus) 
                use a dash in the brainregion section of the file name (i.e. thalamus-CL vs. thalamus-CM).*
        3.	If you want to mask your images to analyze a precise brain region:
            a.	Use Adobe Illustrator or Power Point to overlay the brain atlas to your image and set the boundaries of your brain region of interest.
            b.	Download ImageJ or FIJI if you do not have it installed.  
            *Website to download ImageJ: imagej.net*  
            *Website to download FIJI: fiji.sc*  
            c.	Open the image of interest.
            d.	Select the freehand selection tool on the top menu.        
            e.	Draw the boundaries of the region of interest according to the overlay you created in step 3a. 
            f.	On the top menu select in order:  
                i. Edit > Clear Outside.  
                ii.	Image > Crop.  
            g.	If the signal to noise ratio needs to be enhanced:
                i.	On the top menu select Image > Adjust > Brightness/Contrast  
                ii.	Decrease the “Maximum” value using the sliding bar to increase fluorescence intensity.  
                iii.	Increase the “Minimum” value using the sliding bar to decrease background fluorescence.  
                ***TIP:** Use the histogram (Analyze > Histogram) to confirm that the background fluorescence in the brain region does not contain zero values.*  
            h.	Save the resulting image following AxoDen naming convention.
                
        ## Part 2: Run AxoDen
        4.	Open your web browser and type: axoden.streamlit.app
        5.	Toggle the check box if your images are masked.
        6.	Drag and drop all the files you want to analyze.
        7.	Right after the files have been uploaded, the analysis will run automatically.
        8.	Inspect the control plots to make sure the masking of the image has been detected accurately.
        9.	Download the results by clicking the “Download Results” button of each section.

    ''')

st.set_page_config(
    page_title="Axoden Tutorial & How To",
    page_icon="❔️"
)

how_to_app()
