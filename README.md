# CATCH

Fast radio bursts (FRBs) are mysterious, extragalactic, millisecond-duration, highly energetic radio transient signals. While a majority of FRBs have been detected as isolated events, some sources exhibit a repeating nature. To gain further insights into these mysterious repeating sources, comprehensive analyses of burst properties are often conducted, which make use of machine learning algorithms. However, these algorithms are known to misclassify real FRBs, leading to a discrepancy between the total sample of detected FRBs and the true population. 

## Classification Algorithm and Transient Candidate Handler
A new pipeline is therefore created called the Classification Algorithm and Transient Candidate Handler (CATCH), which classifies candidates based on the `bow tie' feature in the dispersion-measure-time spectrum. This inherent bow tie shape arises due to the dispersive delay and the extragalactic origin from FRBs. CATCH attains a classification sensitivity of 99.62\%, misclassifying merely one real FRB, thereby surpassing the sensitivity of the conventionally employed machine learning algorithms by a factor of 30.

## Code structure
The necessary code to run CATCH is in the CATCH folder:

`box_dm_time.py` is the main file which runs the entire pipeline. It takes information from the completed `eclat` pipeline.

`box_funcs.py` is the file which contains all the functions called in the `box_dm_time.py` file. 

`box_interactive_click.py` is the optional script one can call, which opens a set of figures, in which one can interactively click on certain parts to indicate the bursts. It creates a new CSV containing the locations of the bursts in the spectrum. 

`box_update_bursts.py` runs from the updated CSV, and updates the location, energy, size, and S/N for each burst.

`R117 Analysis Nancay.ipynb` is a notebook containing the complete analysis of FRB 20220912A for the MSc Thesis of D. Konijn.
