# Human Attribute Classification

> **[BMVC 2023] A Comprehensive Crossroad Camera Dataset of Mobility Aid Users**            
> L. Mohr, N. Kirillova, H. Possegger, H. Bischof\
> Paper

> **[TU Graz] Crossroad Camera Dataset - Mobility Aid Users**\
> L. Mohr, N. Kirillova, H. Possegger, H. Bischof\
> [Dataset](https://repository.tugraz.at/records/2gat1-pev27)
 
<img src="docs/exampl_train_data.jpg" title="Exemplary training dataset"/>

## Data Preprocessing
+ Configuration file: `config/opt.py`
+ Extract image patches using yolo bounding box annotations: `data_processing/extract_bboxes.py`
+ Label visualization of extracted patches: `vis_extracted_patches.py`
+ Bounding box visualization on dataset frames (yolo format): `vis_yolo_labels.py` 