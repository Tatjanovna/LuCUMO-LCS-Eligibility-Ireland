# 1.  Overview

This repository contains all supplementary materials associated with the manuscript submitted to PharmacoEconomics Open.
It includes:

- Data preparation scripts
- Transition rate estimation
- Pack-year distribution modelling
- Markov projections
- A master script that automatically executes all notebooks in sequence

All datasets in this project are open and fully reproducible.

## Project Structure

LCS_eligibility_Ireland/
    code/             - Jupyter notebooks and master execution script /
    data_raw/         - Raw input data /
    data_processed/   - Cleaned and harmonised datasets / 
    output/           - Model outputs, figures, and processed projections / 
    supplementary/    - Appendix, data dictionary, diagrams / 

## Important Setup Requirement

a. Before running any notebooks or scripts:  Ensure that the folder "LCS_eligibility_Ireland" is located on your Desktop.

b. All paths in the notebooks and master script are automatically constructed using: Path.home() / "Desktop" / "LCS_eligibility_Ireland"


##  Master Execution Script

A consolidated Jupyter Notebook (run_all_notebooks.ipynb) runs all analysis notebooks in the correct order.
It automatically creates directory paths:

    BASE     = Path.home() / "Desktop" / "LCS_eligibility_Ireland"
    CODE_DIR = BASE / "code"
    RAW_DIR  = BASE / "data_raw"
    PROC_DIR = BASE / "data_processed"
    OUT_DIR  = BASE / "output"

##  To run the full analysis pipeline:

- Open the notebook run_all_notebooks.ipynb located in the code folder
- Run all cells from top to bottom.

This reproduces the entire analysis from raw inputs to final projections.


##   Notebook Execution Order (Handled Automatically)

-cleaning_population_projections.ipynb
-cleaning_smoking_census2022.ipynb
-transition_rates_5y_groups.ipynb
- pack_year_distribution_baseline.ipynb
- eligiblity_estimation.ipynb

##  Path Usage in Notebooks (Portable Structure)

All notebooks now use standardized folder objects:

    RAW_DIR  / "filename.csv"
    PROC_DIR / "filename.csv"
    OUT_DIR  / "figure.png"

Example: Loading a dataset

    df = pd.read_csv(PROC_DIR / "cleaned_smoking_data.csv")

Example: Saving a figure

    filename = fig_title.replace(" ", "_").lower() + ".png"
    plt.savefig(OUT_DIR / filename, dpi=300, bbox_inches='tight')

##  Software Requirements

- Python >= 3.10
- Jupyter Notebook or JupyterLab
- Required Python libraries:
    pandas
    numpy
    scipy
    statsmodels
    matplotlib
    nbconvert
    nbformat
    pathlib
    sklearn
    seaborn
    string
    IPython
    


##  No manual steps are required beyond placing the main folder on the Desktop.


# 2. Related project info 

This project estimates the potential eligibility for Lung Cancer Screening (LCS) in Ireland by integrating smoking trends, pack-year histories, and population projections. It provides insights into screening needs based on smoking prevalence and eligibility trends over time.

## Objectives
1. **Data Integration**: Combine multiple data sources, including:
   - **Census 2022** (smoking prevalence by age, sex, and region).
   - **CSO Population Projections** (demographic forecasts up to 2057).
   - **2017 Eurobarometer Survey** (smoking intensity and history).
   
2. **Descriptive Analysis**: Evaluate regional screening eligibility based on smoking prevalence.

3. **Dynamic Modelling**: Capture the evolution of smoking behavior to estimate future screening eligibility.

## Methodology
A **dynamic modeling approach** is applied to estimate the number of individuals eligible for LCS. Eligibility is determined based on individuals aged **55–75 years** with a smoking history of **≥15 or ≥20 pack-years**. The approach consists of:

### 1. Estimating Pack-Year Distribution
- Uses **Eurobarometer 2017** data to estimate smoking intensity and duration.
- Applies an **exponential decay model** to account for lower smoking rates at younger ages.
- Computes **pack-years** using the formula:

$$
\text{Pack-Years} = \frac{\text{Expected Cigarettes Per Day (ECPD)} \times \text{Smoking Duration (Years)}}{20}
$$

### 2. Modelling Smoking Trends Over Time
- **Markov Process**: Projects future shares of smokers, quitters, and never-smokers.
- Transition probabilities derived from **Eurobarometer data**.
- Annual transition matrix applied to model smoking behavior from **2022 to 2045**.

  **Markov Transition Matrix:**
  \[
  M=
  \begin{bmatrix}
  1 - \text{quit\_rate} & \text{quit\_rate} & 0 \\
  \text{initiation\_rate} & 1 - \text{quit\_rate} & 0 \\
  \text{initiation\_rate} & 0 & 1 - \text{initiation\_rate}
  \end{bmatrix}
  \]

### 3. Estimating Screening Eligibility
- **Log-normal distribution** of pack-years applied to population estimates.
- **Survival function** used to determine screening eligibility proportions.
- Baseline screening eligibility calculated using **Census 2022 smoking prevalence**.

### 4. Projecting Pack-Year Trends Over Time
- Pack-years **discounted** over time to reflect:
  - Lower smoking intensity in younger cohorts.
  - Increasing cessation rates.
  - Policy-driven reductions in smoking prevalence.

## Data Sources
| Source | Description |
|--------|------------|
| **Census 2022 (Ireland)** | Provides baseline population data and smoking prevalence by age, sex, and region. |
| **CSO Population Projections** | Forecasts population counts up to 2057, considering aging and mortality trends. |
| **Eurobarometer 2017** | Individual-level smoking data, including initiation, intensity, and cessation rates. |



