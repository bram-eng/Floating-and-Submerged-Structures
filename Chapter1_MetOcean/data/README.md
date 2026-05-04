# MetOcean Data

Place your MetOcean data CSV files in this directory.

## Expected Format

Each CSV file should have the following columns:

| Column     | Description                        | Unit   |
|------------|------------------------------------|--------|
| `datetime` | ISO-8601 timestamp                 | –      |
| `Hs`       | Significant wave height            | m      |
| `Tp`       | Peak spectral wave period          | s      |
| `U10`      | Wind speed at 10 m above sea level | m/s    |
| `Vc`       | Surface current speed              | m/s    |

### Example

```csv
datetime,Hs,Tp,U10,Vc
2000-01-01 00:00,1.23,8.4,7.2,0.15
2000-01-01 03:00,1.45,9.1,8.5,0.18
...
```

## Data Sources

Recommended freely available reanalysis products for the Gulf of Finland:

- **ERA5** – ECMWF Reanalysis v5 (hourly, 0.25° grid)
  <https://cds.climate.copernicus.eu>
- **CMEMS** – Copernicus Marine Service Baltic Sea Physics Reanalysis
  <https://marine.copernicus.eu>

Download the data for the site coordinates (approx. 59.8 °N, 25.0 °E for the
Helsinki–Tallinn crossing) and export as CSV using the scripts provided in the
parent directory.

## Sample Data

`sample_gulf_of_finland.csv` – Synthetic 30-year hourly record generated with
`generate_sample_data.py` for testing and demonstration purposes.
Do **not** use this synthetic data for final design values.
