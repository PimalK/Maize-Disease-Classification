🌽 Comprehensive Maize Disease Classification via Deep Learning with Integrated Multi-Source Data

📌 Authors

Pimal Khanpara, Bhaumik Panchal, Preeti Kathiria, Usha Patel, Deekshita Athreya**(* denotes corresponding authorship)

📜 Abstract

Accurate detection of maize diseases is crucial for ensuring food security and agricultural productivity. However, existing datasets for maize disease detection often lack diversity, typically relying on either lab-generated or field-captured images, and covering a limited range of disease classes.

To address these limitations, we present a study that integrates three distinct data sources:

🌱 PlantVillage (lab-generated images)

🌾 Tanzania (field-captured images)

🚁 NLB (drone-mapped images)

This combined dataset increases the number of disease classes while also enhancing the robustness and generalizability of the trained models. We applied seven popular deep learning (DL) models, including AlexNet, VGG16, CNN, MobileNet, ResNet50, DenseNet121, and Xception, leveraging transfer learning for all models except the custom CNN.

Our comparative performance analysis reveals significant improvements in model accuracy and robustness attributed to the dataset’s enhanced diversity. Among the tested models, 🏆 DenseNet121 achieved the highest average accuracy of 97.28%, underscoring its superior ability to generalize across diverse image sources.

This study highlights the potential of multi-source data integration and transfer learning in building reliable maize disease classification systems, offering valuable insights into the strengths and limitations of different DL approaches in practical agricultural applications.

⚙️ Installation

Clone the repository and install the required dependencies:

pip install -r requirements.txt

📂 Dataset Preparation

A comprehensive and diverse dataset for maize disease detection was created by combining images from three distinct datasets: PlantVillage, Tanzania, and NLB.

📁 Dataset Structure

- Combined
  - 🌿 HEALTHY
  - 🍂 Maize_Blight
  - 🍁 Maize_Common_Rust
  - 🍃 Maize_Gray_Leaf_Spot
  - 🌾 MLN
  - 🌱 MSV

🧠 Models Used

The following deep learning models were implemented:

⚡ Xception

🔍 VGG16

🛠️ ResNet50

📱 MobileNet3D

📊 DenseNet121

🖥️ AlexNet

📝 Custom CNN

📊 Results & Performance

Among all tested models, 🏆 DenseNet121 achieved the highest accuracy of 97.28%, demonstrating superior generalization across diverse image sources.

📜 License

This project is open-source and available under the MIT License.

📧 Contact

For any inquiries, feel free to contact the corresponding authors:

📩 Preeti Kathiria

📩 Usha Patel

🤝 Contribution

We welcome contributions! Feel free to fork the repository, raise issues, or submit pull requests to enhance the project.

