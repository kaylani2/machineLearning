## Files:

### UNSW_2018_IoT_Botnet_Full5pc_{1,2,3,4}.csv

Five percent of the original dataset. Labels are not balanced. Some files don't have all labels (attempting to read them will throw errors).


### UNSW_2018_IoT_Botnet_Dataset_{1..74}.csv

Full dataset. Labels are not balanced.

**Full dataset files do not have a header with the features. It is necessary to import them from the features file.**

### UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv

Features file.

## Directories:

### attack_classification/:

Multiclass classification.

### attack_identification/:

Binary classification.

## Note:

No pipelines are used for model evaluation. No cross-validation, the data is processed manually after splitting the dataset. **If pipelines are used and cross-validation is performed, the file will have a CV suffix.** Like: mlp_CV.py

Unless specified otherwise, the code uses the 5% dataset.
