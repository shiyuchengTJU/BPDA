# Source-free and Black-box Domain Adaptation via Distributionally Adversarial Training (Pattern Recognition 2023)
Official implementation of ["**Source-free and Black-box Domain Adaptation via Distributionally Adversarial Training (PR 2023)**"], Yucheng Shi, Kunhong Wu, Yahong Han, Yunfeng Shao, Bingshuai Li, Fei Wu.

> **Abstract:** *Source-free unsupervised domain adaptation is one class of practical deep learning methods which generalize in the target domain without transferring data from source domain. However, existing source-free domain adaptation methods rely on source model transferring. In many data-critical scenarios, the transferred source models may suffer from membership inference attacks and expose private data. In this paper, we aim to overcome a more practical and challenging setting where the source models cannot be transferred to the target domain. The source models are considered as queryable black-box models which only output hard labels. We use public third-party data to probe the source model and obtain supervision information, dispensing with transferring source model. To fill the gap between third-party data and target data, we further propose Distributionally Adversarial Training (DAT) to align the distribution of third-party data with target data, gain more informative query results and improve the data efficiency. We call this new framework Black-box Probe Domain Adaptation (BPDA) which adopts query mechanism and DAT to probe and refine supervision information. Experimental results on several domain adaptation datasets demonstrate the practicability and data efficiency of BPDA in query-only and source-free unsupervised domain adaptation.*

## 1. Settings
## 1.1 Different settings of domain adaptation:
<p align="center">
    <img src='/setting.png' width=900/>
</p>

**(a) Supervised Domain Adaptation:**
- Source and target data in a same node
- Labeled target data

**(b) Unsupervised Domain Adaptation:**
- Source and target data in a same node
- Unlabeled target data

**(c) Source-free Unsupervised Domain Adaptation:**
- Source and target data can be in different node
- Unlabeled target data



## 1.2 Our setting: **Query-only and Source-free Unsupervised Domain Adaptation**

This is the **first exploration of black-box source-free UDA setting** that **source and target domain models cannot be transferred**.

<p align="center">
    <img src='/our setting.png' width=900/>
</p>
The red lock signs on dataset and model indicate that they cannot be accessed by target domain. The setting of (b) in this paper can only access the unlabeled target data and hard labels obtained by querying without target data. The simultaneous absence of source data, target data, source model and logits information prevents potential membership inference attack.

## 2. Method

<p align="center">
    <img src='/framework.png' width=900/>
</p>

**(a) Train the source model:**
- Use label smoothing technique

**(b) Black-box Initialization with Third-party Dataset:**
- Input any image into source model, it can output a classify result, which contains the model information. 

**(c) Distributionally Adversarial Training:**
- Fine-tune target model with unlabeled target data.
- Generate adversarial examples and retrain the target model.

## 3. Usage
### 3.1 Prepare data
The datasets used in the paper are available at the following links:
* [Imagenet](http://image-net.org/index)
* [Digit-5](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#digit-5)
* [DomainNet](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#domainnet)
* [VisDA](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#visda17)
* [Office-31](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#office-31)
* [Office-Home](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#office-home)

<p align="center">
    <img src='/data.png' width=900/>
</p>

### 3.2 Function of each script

- train_source.py: Train a model for a single source domain and test its performance on other domains after training
- train_target.py: Fine-Tune the code in the target domain and extract the pre-trained source domain model for further adjustment. After the extracting process, you only need to adjust the model load path in this code
- train_extract.py: Based on the idea of Model Extraction Attack, a third-party data set is used to probe the input-output relationship of a black box model, thereby extracting the model. The code simply steals the model without doing anything related to the target domain, so the stolen model can be tuned on multiple target domains. The code is saved to the model with the best average performance of each target domain, but in the actual scenario, the target domain data cannot be used to test during the theft process.
- train_target_more_finetune.py: Fine-Tune the code in the target domain and extract the pre-trained source domain model for further adjustment. After the extracting process, you only need to adjust the model load path in this code.
- test_target.py: Test the performance of the final target domain model
- test_extract.py: Test the performance of the extract model on various domains
- test_source.py: Test the performance of the aggregate model or individual model on each target domain in the target domain list