# Floating and Submerged Structures – Helsinki–Tallinn Tunnel

University group project for the design of a **Floating Submerged Tunnel (FST)** across the Gulf of Finland between Helsinki (Finland) and Tallinn (Estonia).

---

## Project Overview

The tunnel is designed as a submerged floating tunnel (SFT) anchored below the sea surface. The project covers the full engineering design lifecycle, from site characterisation and extreme environmental loading through to structural reliability assessment.

---

## Repository Structure

```
.
├── Chapter1_MetOcean/          # MetOcean data & Extreme Value Analysis
│   ├── data/                   # Raw CSV data files (waves, wind, current)
│   ├── metocean_analysis.py    # Data loading and statistical summaries
│   ├── extreme_value_analysis.py  # EVA: GEV / GPD fitting & return periods
│   └── plots.py                # Visualisation utilities
│
├── Chapter2_ConceptSelection/  # Concept selection & Life Cycle Analysis
│   └── concept_selection.ipynb
│
├── Chapter3_MooringDesign/     # Mooring system design
│   └── mooring_design.ipynb
│
├── Chapter4_NumericalModelling/ # Numerical modelling & dynamical assessment
│   └── numerical_modelling.ipynb
│
├── Chapter5_Reliability/       # Reliability & probabilistic assessment
│   └── reliability_assessment.ipynb
│
├── requirements.txt
└── README.md
```

---

## Chapters

### Chapter 1 – MetOcean Data & Extreme Value Analysis
Collection and processing of MetOcean data (significant wave height *Hs*, peak wave period *Tp*, wind speed *U10*, current speed *Vc*) from ERA5 / CMEMS reanalysis products. Extreme Value Analysis using the Block Maxima (GEV) and Peaks-Over-Threshold (GPD) methods to derive design values for specified return periods (10, 25, 50, 100 years).

### Chapter 2 – Concept Selection & Life Cycle Analysis
Evaluation of structural concepts (pontoon-supported, tether-moored, hybrid) against technical, economic, environmental, and risk criteria. Multi-criteria decision analysis (MCDA) and a high-level life cycle assessment (LCA).

### Chapter 3 – Mooring Design
Quasi-static and dynamic analysis of the tether / anchor system. Catenary and taut-leg configurations, mooring line sizing, and fatigue screening.

### Chapter 4 – Numerical Modelling & Dynamical Assessment
Frequency-domain and time-domain dynamic analysis of the tunnel structure subjected to wave, current, and seismic loading. Modal analysis, transfer functions, and extreme response estimation.

### Chapter 5 – Reliability & Probabilistic Assessment
Structural reliability analysis using First-Order Reliability Method (FORM) and Monte Carlo Simulation. Computation of annual failure probabilities and reliability indices (β) for governing limit states.

---

## Getting Started

### Prerequisites
Python ≥ 3.9 is required. Install all dependencies with:

```bash
pip install -r requirements.txt
```

### Running the MetOcean Analysis
```bash
cd Chapter1_MetOcean
python metocean_analysis.py        # Statistical summary of MetOcean data
python extreme_value_analysis.py   # EVA and return-period estimates
```

Place your raw CSV data files in `Chapter1_MetOcean/data/`. See `metocean_analysis.py` for the expected column format.

---

## Contributors
- Group project – TU Delft / [University Name]

## License
For academic use only.
