[![DOI](https://zenodo.org/badge/928787433.svg)](https://doi.org/10.5281/zenodo.14831784)
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
![flowchart](https://github.com/user-attachments/assets/fbd2574d-a48c-454b-bc69-4091297eabae)  

**Flow of proposed approach**

---

## Installation
The code has the following dependencies:
- `numpy`
- `pickle-mixin`
- `opencv-python`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `keras`
- `pillow`

Ensure you have Python installed, then install the required dependencies:
```sh
pip install -r requirements.txt
```

## Dataset Preparation
A comprehensive and diverse dataset for maize disease detection was created by combining images from three distinct datasets. 1000 images of each class were randomly chosen from 23,655 images in the combination. For classes with less than 1000 images, augmentation was performed.
### Dataset Links

- **PlantVillage Dataset**  
  [Corn/Maize Leaf Disease Dataset on Kaggle](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset)

- **Maize Imagery Dataset - Tanzania**  
  [Mendeley Data Repository](https://data.mendeley.com/datasets/fkw49mz3xs/1)

- **Corn Leaf Diseases (NLB)**  
  [Corn Leaf Diseases (NLB) on Kaggle](https://www.kaggle.com/datasets/rabbityashow/corn-leaf-diseasesnlb)

### Dataset Structure:
```
- Dataset
  - HEALTHY
  - Maize_Blight
  - Maize_Common_Rust
  - Maize_Gray_Leaf_Spot
  - MLN
  - MSV
```

## Models Used
We experimented with the following deep learning models. All the pretrained models were fine-tuned on the combined dataset specified above.

- **Xception**: Xception replaces Inception modules with depthwise separable convolutions and residual connections, which can be expressed as:  
  **y = FDW(FPW(x))**  
  where **FDW** is the depthwise convolution and **FPW** is the pointwise convolution.  
  This structure reduces parameters while improving performance in image classification.

- **VGG16**: VGG16 is a 13-layer CNN that utilizes ReLU activation, max pooling, and fully connected layers for classification. The convolutional layers apply filters **W** to input **X**, producing feature maps:  
  **Z = max(0, W * X + b)**  
  Max pooling is used to reduce spatial dimensions, and the fully connected layers combine features as:  
  **z(l) = max(0, W(l) * z(l−1) + b(l))**  
  The final output is generated using a softmax function to obtain class probabilities.

- **ResNet50**: ResNet solves the vanishing gradient problem using skip connections (residual connections), which can be mathematically represented as:  
  **H(x) = F(x) + x**  
  where **H(x)** is the output of the residual block, **F(x)** is the function to learn, and **x** is the input.  
  This structure helps in improving gradient flow in deep networks.

- **MobileNetV2**: MobileNet is optimized for mobile devices, employing depthwise separable convolutions, which can be defined mathematically as:  
  **y = D(x) * S(x)**  
  where **D** denotes the depthwise convolution and **S** denotes the pointwise convolution (1x1 convolution).  
  This architecture uses adjustable parameters to balance accuracy with efficiency.

- **DenseNet121**: DenseNet connects each layer to every other layer, enhancing gradient flow and feature reuse. The output for a layer in DenseNet121 can be represented as:  
  **xl = H(xl−1, xl−2, ... , x0)**  
  where **xl** is the output of layer **l** and **H** is the composite function of the operations performed.  
  Dense blocks and transition layers are utilized for efficient learning.

- **AlexNet**: AlexNet is a CNN with five convolutional layers, using ReLU activation, max pooling, Local Response Normalization (LRN), and dropout to prevent overfitting. The architecture can be represented mathematically as:  
  **y = f(W · x + b)**  
  where **y** is the output, **W** is the weight matrix, **x** is the input, **b** is the bias, and **f** is the ReLU activation function.  
  It is trained with backpropagation using the RMSprop optimizer:  
  **W ← W − η∇L**  
  where **η** is the learning rate and **L** is the loss function.

- **Custom CNN**: The customized CNN is tailored for specific tasks and consists of convolutional and pooling layers, batch normalization, dropout, and fully connected layers to map features to output classes.  
  The output of a convolutional layer can be represented as:  
  **z = σ(W * x + b)**  
  where **z** is the output, **σ** is the activation function (e.g., ReLU), and **W * x** represents the convolution operation.

<img width="611" alt="CNN architecture" src="https://github.com/user-attachments/assets/8faa9596-f2e2-465f-bbb3-de5ea33fbf2c" />  

**Architecture of custom CNN**

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/maize-disease-classification.git
   cd maize-disease-classification
   ```
2. Each ipynb file contains the complete code of the model specified in the filename.
3. Run the cells of each ipynb file in sequence to replicate the results.
4. The trained models will be saved to the root directory with a .h5 extension.
---

## Results
- **Best Performing Model:** DenseNet121
- **Highest Accuracy:** 97.28%

## Cite this repository as
```sh
PimalK, “PimalK/Maize-Disease-Classification: Maize Disease Classification”. Zenodo, Feb. 07, 2025. doi: 10.5281/zenodo.14831785.
```
## Cite the manuscript as
```sh
@article{Khanpara2025MaizeDisease,
  author    = {Pimal Khanpara and Bhaumik Panchal and Preeti Kathiria and Usha Patel and Deekshita Athreya},
  title     = {Comprehensive Maize Disease Classification via Deep Learning with Integrated Multi-Source Data},
  journal   = {The Visual Computer},
  year      = {2025},
  publisher = {Springer}
}

```
**For more details, refer to the manuscript or contact the corresponding authors.**
