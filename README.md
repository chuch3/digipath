# digipath 
 
### Problem definition

Why apply digital pathology analysis instead of MRI for tumors? 

Whole-slide images provides morphological biomarkers (tissue ROIs) and cellular resolution,
capturing finer details of each tissue sample over subjective visual indicators from MRIs.
Giving more precise insights to cancer / tumor diagnosis and grading in AI/ML applications.

Digital pathology is proven to improve workflows by 13% and clear return on investments (ROIs), 
establishing cost advantage over traditional MRI consultations and providing lab revenue.

[(Wilson et al., 2017)](https://clpmag.com/diagnostic-technologies/digital-pathology/digital-pathology-gives-rise-computational-pathology/)

So moving on to this project...! :(


### Dataset Description

- LungHist700 [(Diosdado et al., 2024)](https://doi.org/10.1038/s41597-024-03944-3)
    - "Accurate detection and classification of lung malignancies are crucial for 
    early diagnosis, treatment planning, and patient prognosis." 
    - "... a dataset of 691 high-resolution (1200 × 1600 pixels) histopathological lung images, covering adenocarcinomas, squamous cell carcinomas, and normal tissues from 45 patients. These images are subdivided into three differentiation levels for both pathological types: well, moderately, and poorly differentiated, resulting in seven classes for classification. The dataset includes images at 20x and 40x magnification, reflecting real clinical diversity. "

### Dataset Setup 

Download the dataset [online](https://figshare.com/articles/dataset/LungHist700_A_Dataset_of_Histological_Images_for_Deep_Learning_in_Pulmonary_Pathology/25459174), extract it and place onto the `data/` folder for analysis.

## Todo

Start : 2026-05-11 

- [ ] WSI image analysis 
- [ ] Select and apply model for WSI-specific applications

## Ideas

- Multiple Instance Learning 
