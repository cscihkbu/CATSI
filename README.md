# CATSI: Context-Aware Time Series Imputation

This repository contains the source codes of the CATSI model, a submission to the 2019 ICHI Data Analytics Challenge on Missing data Imputation (DACMI).

## Citations
If you find the paper or the implementation helpful, please cite the following papers:
```bib
@article{yin2020context,
  title={Context-Aware Time Series Imputation for Multi-Analyte Clinical Data},
  author={Yin, Kejing and Feng, Liaoliao and Cheung, William K},
  journal={Journal of Healthcare Informatics Research},
  year={2020},
  publisher={Springer}
}
```
```bib
@inproceedings{yin2019context,
  title={Context-Aware Imputation for Clinical Time Series},
  author={Yin, Kejing and Cheung, William K},
  booktitle={2019 IEEE International Conference on Healthcare Informatics (ICHI)},
  year={2019},
  organization={IEEE}
}
```

## Requirements
The model is tested with:
```
Python 3.7.3
PyTorch 1.0.1
numpy 1.16.2
pandas 0.24.2
tqdm 4.31.1
```

## Running the codes
Run the following command:
```bash
python main.py -i /path/to/training/data -t /path/to/test/data
```
