# DPNN_Kidney
#https://doi.org/10.5281/zenodo.16782208



The growing prevalence of malignant kidney tumors poses significant challenges in patient care, particularly in accurate classification and treatment. 
Traditional computer-aided diagnostics, such as percutaneous needle biopsy, considered the gold standard, can pose risks and introduce internal heterogeneity. 
The complex tissue structure, blurred boundaries, surrounding fat and blood vessels, irregular morphology, and varying lesion sizes in renal CT images often lead to false diagnoses and reduced accuracy.
To address these challenges, this study proposes a robust diagnostic framework using a novel Dual Path Neural Network (DPNN) that combines deep semantic learning with radiomic techniques for precise renal tumor classification.  
The proposed algorithm enhances tumor prognosis accuracy and robustness, minimizes human bias, and reduces overfitting through sophisticated preprocessing, filtering, and data augmentation techniques, thereby strengthening findings' robustness of findings. 
Leveraging a dataset of 12,446 CT images categorized into cyst, stone, tumor, and normal, the proposed DPNN algorithm extracts both global contextual and local fine-grained features through parallel deep and shallow learning pathways. The architecture integrates spatial pyramid pooling, dropout regularization, and batch normalization, supported by transfer learning from five advanced CNN architectures (mVGG19, InceptionV3, ResNet152V2, EfficientNetB7, and EANet). Additionally, radiomic features such as texture and shape descriptors are fused to enhance predictive accuracy. 
The proposed DPNN demonstrates high accuracy for renal tumor classification based on a publicly available dataset.
It achieves the highest accuracy (99.2\%), F1-Score (0.995), AUC (0.995), and specificity (0.993), while maintaining the lowest MAE (0.010), MSE (0.005), and RMSE (0.069). Compared to the next-best model, EANet, DPNN improves accuracy by 0.6\%, F1-Score by 1.0\%, AUC by 1.4\%, and reduces RMSE by 14.8\%. 
Furthermore, DPNN is computationally efficient, with the fastest training (19ms) and inference speed (12ms per image). The findings indicate a promising direction for future work to validate the model's performance with more diverse datasets, ultimately supporting its real-world application in clinical settings.


Dataset Acquisition
The dataset used in this study is a modified version of the publicly available kidney CT image dataset from Kaggle \footnote{\url{https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone}} \hl{which has undergone preprocessing by third-party sources. The dataset consists of 12,446 CT images, categorized into cyst, stone, tumor, and normal conditions. The images were processed to standardize dimensions, reduce noise, and enhance image quality for better feature extraction. These steps ensured that the dataset was standardized for deep learning applications and improved the model's generalization capability.}
The dataset contains coronal and axial image cuts obtained through contrast and non-contrast investigative procedures on the whole abdomen and urogram specifications. 
Coronal slices offer a wide-angle view of kidney anatomy, enabling the distinction between various diseases. Axial slices provide close evaluations of such smaller anatomical structures, which help trace the slightest pathological changes. The DPNN in our study leverages the complementary qualities between the two modalities of imaging to enhance the accuracy of diagnosis.  This approach captures a full spectrum of pathological features necessary for precise cancer classification, thereby improving the model's robustness and generalization ability across unseen data. 

This dual-view method significantly outperforms single-view approaches in classifying renal diseases.
Collecting data entails a meticulous selection of DICOM investigations, wherein specific diagnoses are isolated to generate distinct DICOM image batches corresponding to each radiological discovery. Patient privacy is preserved by omitting crucial information and metadata from the DICOM images, which are transformed into a lossless JPEG format. 
As a consequence of this rigorous methodology, our dataset consists of 12,446 unique entries, which have been meticulously classified into 3,709 instances of cysts, 5,077 cases of standard samples, 1,377 examples of stones, and 2,283 occurrences of tumors. 
The class labels for each image (cyst, normal, stone, or tumor) were manually assigned by expert radiologists based on confirmed clinical diagnoses at the time of imaging. Labeling was carried out during the data curation process using the metadata embedded within the hospital. The images were only included if they had a clear and confirmed diagnosis, which ensured the reliability of the ground truth used for model training and evaluation.


The dataset for a kidney stone model may introduce biases due to its homogeneity, particularly in the similarities in shape and texture of kidney stones. This could impact the model's ability to classify new data accurately and may lead to performance degradation in real-world clinical settings. To mitigate overfitting, various data augmentation techniques were applied to enhance the diversity and variability of training images. However, the dataset may not fully represent the diversity of kidney conditions encountered in clinical practice, especially rare or atypical cases. 
The kidney image dataset is meticulously organized and distributed into four classes, including comprehensive training, testing, and validation sets, as described in \hyperlink{t01aa}{Table \ref{t01aaa}}.

\begin{table*}[ht!]
\centering
\caption{The kidney image collection before data augmentation is organized into four classes: training, testing, and validation.} \label{t01aaa}
\begin{tabular}{p{15mm}p{25mm}p{25mm}p{25mm}p{25mm}} \hline
Class & Total Images & Training Data  & Validation Data & Test Data \\ \hline
Cyst  & 3709 &  2967 & 371 & 371 \\ \hline 
Normal& 5077 &  4061 & 508 & 508 \\ \hline
Stone & 1377 &  1101 & 138 & 138 \\ \hline
Tumor & 2283 &  1827 & 228 & 228 \\   \hline
Total & 12,446 &  9956 & 1245 & 1245 \\   \hline
\end{tabular}
\end{table*}
