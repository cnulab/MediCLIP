### Datasets Links  
- **BUSI  [[Baidu Cloud (pass8866)]](https://pan.baidu.com/s/1EVt96fExiqrvMQslPDRRRg?pwd=8866)   [[Google Drive]](https://drive.google.com/file/d/1PyvMXdNEVY86BY1PV8yKhPVS30TAmS6X/view?usp=drive_link)  [[Official Link]](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)**  
- **BrainMRI  [[Baidu Cloud (pass8866)]](https://pan.baidu.com/s/1--5vPMN-eTqePPYjpKTwvA?pwd=8866)  [[Google Drive]](https://drive.google.com/file/d/1kldE-5_wXaN-JR_8Y_mRCKQ6VZiyv3km/view?usp=drive_link)  [[Official Link]](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)**  
- **CheXpert [[Baidu Cloud (pass8866)]](https://pan.baidu.com/s/15-V5wobA_7ICvZAXBraDGA?pwd=8866)  [[Google Drive]](https://drive.google.com/file/d/1pVYRipGC2VqjYP-wHdDFR-lLf7itLiUi/view?usp=drive_link)  [[Official Link]](https://stanfordmlgroup.github.io/competitions/chexpert/)**  

### The complete directory structure is as follows:
```
    |--data                         
        |--brainmri           
            |--samples
                |--train.json
                |--test.json
            |--images
                |--test
                    |--abnormal
                       |--image_97.jpg
                       |--...
                    |--normal
                       |--image_32.jpg
                       |--...
                |--train
                    |--normal
                       |--image_0.jpg
                       |--...
        |--busi           
            |--samples
                |--train.json
                |--test.json
            |--images
                |--test
                    |--abnormal
                       |--benign_0.jpg
                       |--...
                    |--normal
                       |--normal_32.jpg
                       |--...
                    |--ground_true
                       |--benign_mask_0.jpg
                       |--...                    
                |--train
                    |--normal
                       |--normal_0.jpg
                       |--...
        |--chexpert           
            |--samples
                |--train.json
                |--test.json
            |--images
                |--test
                    |--abnormal
                       |--00002.jpg
                       |--...
                    |--normal
                       |--00960.jpg
                       |--...
                |--train
                    |--normal
                       |--00000.jpg
                       |--...
```
