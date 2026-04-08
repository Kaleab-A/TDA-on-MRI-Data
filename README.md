# MRI Project

## Environment and Reproducibility

This project was tested with:

* Python 3.11
* macOS `osx-64`
* Conda with the `conda-forge` channel

Most dependencies install cleanly through `conda-forge`.

`giotto-tda` is not available from the conda channels used for this project, so it must be installed separately with `pip` **after** creating and activating the conda environment.

## What someone else needs to reproduce this

To reproduce this environment, a user needs:

1. Conda installed
2. The `environment.yml` file from this repository
3. Python 3.11
4. The extra `pip` install for `giotto-tda`

They should not need to know the troubleshooting history or any hidden install order beyond the documented setup steps below.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate mri_project
```

### 3. Install the extra package not available in conda

```bash
pip install giotto-tda==0.6.2
```

## Verification

To verify that the environment is working, run:

```bash
python -c "import numpy, pandas, scipy, sklearn, nilearn, ripser, persim, gudhi, skfda; print('core imports OK')"
```

If you also installed `giotto-tda`, run:

```bash
python -c "import gtda; print('giotto-tda OK')"
```

## Notes

* This project uses **conda** for the main environment because some packages with compiled dependencies were difficult to install reliably with `pip` alone.
* `giotto-tda` is installed separately with `pip` because it was not available from the conda channels used here.
* This setup was tested on **Python 3.11**. Other Python versions may fail for some dependencies.

## Files to include in the repository

At minimum, include:

* `README.md`
* `environment.yml`

Optional but helpful:

* `.python-version` to indicate Python 3.11 clearly
* `requirements.txt` only if you want to track additional pip-only packages

## Example repository structure

```text
.
├── README.md
├── environment.yml
└── src/
```

## Example `environment.yml`

```yaml
name: mri_project
channels:
  - conda-forge
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scipy
  - scikit-learn
  - nilearn
  - ripser
  - persim
  - gudhi
  - scikit-fda
  - matplotlib
  - seaborn
  - networkx
  - plotly
  - tqdm
  - joblib
  - jupyterlab
  - ipywidgets
  - pip
```
