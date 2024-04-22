# PepCA
 Unveiling Protein-Peptide Interaction Sites with a Multimodal Neural Network Model

## Installation
Clone the repository and Install requirements
```bash
git clone https://github.com/cloudaner115/PepCA.git
cd PepCA
pip insatll -r requirements.txt
```

You can use this one-liner for installation, using the latest release of esm:
```bash
pip install fair-esm  # latest release, OR:
pip install git+https://github.com/facebookresearch/esm.git  # bleeding edge, current repo main branch
```

## Training PepCA

Before training PepCA, you need to run the `python run_esm.py` command to ensure that the protein sequence esm encoding has been run and saved to the pkl file.

To train PepCA run the following command.

```bash
python train.py -d pepnn 
```

`-d` specifies the dataset to train on, can be set to either pepnn, pepbind, bitenet or interpep
