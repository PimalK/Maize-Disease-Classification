# Comprehensive Maize Disease Classification via Deep Learning with Integrated Multi-Source Data

**Authors:** Pimal Khanpara, Bhaumik Panchal, Preeti Kathiria*, Usha Patel*, Deekshita Athreya  
(* denotes corresponding authorship)

## Abstract
Accurate detection of maize diseases is crucial for ensuring food security and agricultural productivity. However, existing datasets for maize disease detection often lack diversity, typically relying on either lab-generated or field-captured images, and covering a limited range of disease classes. 

To address these limitations, we present a study that integrates three distinct data sources:
- **PlantVillage** (lab-generated images)
- **Tanzania** (field-captured images)
- **NLB** (drone-mapped images)

This combined dataset not only increases the number of disease classes but also enhances the robustness and generalizability of the trained models.

We applied seven popular deep learning (DL) models, including AlexNet, VGG16, CNN, MobileNet, ResNet50, DenseNet121, and Xception, leveraging transfer learning for all models except the custom CNN. Our comparative performance analysis reveals significant improvements in model accuracy and robustness attributed to the dataset’s enhanced diversity. 

Among the tested models, **DenseNet121 achieved the highest average accuracy of 97.28%**, underscoring its superior ability to generalize across diverse image sources. 

This study highlights the potential of multi-source data integration and transfer learning in building reliable maize disease classification systems, offering valuable insights into the strengths and limitations of different DL approaches in practical agricultural applications.

---

## Installation
Ensure you have Python installed, then install the required dependencies:
```sh
pip install -r requirements.txt
```

## Dataset Preparation
A comprehensive and diverse dataset for maize disease detection was created by combining images from three distinct datasets.
### Dataset Links

- **PlantVillage Dataset**  
  [Corn/Maize Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)

- **Maize Imagery Dataset - Tanzania**  
  [Mendeley Data Repository](https://data.mendeley.com/datasets/fkw49mz3xs/1)

- **Corn Leaf Diseases (NLB)**  
  [Corn Leaf Diseases (NLB) on Kaggle](https://www.kaggle.com/datasets/rabbityashow/corn-leaf-diseasesnlb)


### Dataset Structure:
```
- Combined
  - HEALTHY
  - Maize_Blight
  - Maize_Common_Rust
  - Maize_Gray_Leaf_Spot
  - MLN
  - MSV
```

## Models Used
We experimented with the following deep learning models:
- **Xception**
- **VGG16**
- **ResNet50**
- **MobileNet3D**
- **DenseNet121**
- **AlexNet**
- **Custom CNN**

## Results
- **Best Performing Model:** DenseNet121
- **Highest Accuracy:** 97.28%

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/maize-disease-classification.git
   cd maize-disease-classification
   ```
2. Run the ipynb files in models.zip as required
---
**For more details, refer to the documentation or contact the corresponding authors.**
