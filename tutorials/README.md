# OPM-MEG Tutorials in osl-ephys

### The tutorials below are currently being updated. 

### ⚙️ Software requirements

These tutorials rely on **homogeneous field correction**  
(Tierney et al., 2021), which is only available in:

- **`MNE` ≥ 0.18**
- **`osl-ephys` ≥ 0.24**

Please ensure your environment is up to date before continuing.

<img src="20250107_12_27_06_01.jpg" style="width:30%;">

## Download the sample data from osf (sub-001)

```python
import os

basedir = os.getcwd()

def get_data(outdir, zip_name="study-FourMotorOPM.zip"):
    """Download and unzip the dataset from OSF project xh9b6."""
    if os.path.exists(os.path.join(outdir, zip_name.replace(".zip", ""))):
        print("Data already exists. Skipping download.")
        return

    os.system(f"osf -p xh9b6 fetch {zip_name}")
    os.system(f"unzip -o {zip_name} -d {outdir}")
    os.remove(zip_name)

    print(f"Data downloaded and extracted to: {outdir}")

get_data(basedir)
```

#### Note - more subjects will be uploaded soon

## 1. [Preprocessing](01_OPM_preprocessing_forOSLcourse.md)
## 2. [Sensor-Level Analysis (Time-Frequency)](02_sensor_level_TFR.md)
## 3. [Coregistration](03_rhino_coreg.md)
## 4. [Source Localisation](04_source_recon.md)

## 5. [Parcellation](05_parcellation.md)
