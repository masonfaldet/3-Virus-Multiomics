Barb, I am sharing a folder with the 3 Virus data. 


3VLP = 3 Virus Lipidomics Positive
3VLN = 3 Virus Lipidomics Negative
3VMP = 3 Virus Metabolomics Positive
3VMN = 3 Virus Metabolomics Negative

The '250708_3VLP_heatmaps.rmd' file I am sending to provide you with:
    > colorways used for lipids classes, treatments, viruses, DPI, etc
    >the factor order for plotting lipid classes
    >the factor order for plotting treatments etc
    >the code that I used to generate the 'data' df that I generally work with including virus titers,           metadata, etc.
    >how I combine lipid maps IDs to peaklist, and generate a single feature-LMID match by filtering for     the lowest Delta 

The .rmd files named like 'dataPreprocessing_Project.rmd' contain:
    >read plist and generate metadata (there are some nuances to each peak list because of different     naming schemes or misnamed files — so it may be worth your time to take my code into account     rather than finding these nuances yourself)
    >group filtering and 2 step imputation
    >limma analysis
    >lipidannotator MSMS identification results
    >lipidmaps tentative identification results

The metadata folder contains all of the metadata that you may or may not want to use I've generated such as:
    >tentative identification searches (generally lipidmaps 'LMID' for lipids, and CEU mass mediator     search for metabolomics)
    >full database search results in their own folder
    >lipidannotator MSMS identification results
    >shortened list of metadata without full plist info
    >limma results


This should get you started. Let me know if you need anything else or have questions.
Paul