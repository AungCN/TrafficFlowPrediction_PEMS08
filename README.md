# Traffic Flow Prediction on PeMS08 Dataset

Comparative time-series analysis and deep learning forecasting of traffic flow, occupancy, and speed on the Caltrans Performance Measurement System (**PeMS08**) dataset using **R** (ARIMA, Exponential Smoothing, Prophet, and Keras/TensorFlow neural networks).

## Repository Contents

- `TrafficFlow_PESM08DataSet_V2.R` — Complete R implementation for PeMS08 exploratory data analysis, stationarity testing, seasonal decomposition, classical time-series models, Prophet, and deep learning forecasting.
- `traffic.csv` — PeMS08 traffic sensor dataset (flow, occupancy, speed across sensors and time steps).
- `traffic_flow_pems08_comparative_analysis_doc_SN_Origin.docx` — Comparative analysis research manuscript.
- `MSDS511_Timeseries_PPTX_20241206 9.08.01 AM.pdf` — Presentation slides summarizing methodology and experimental results.
- `Traffic-Flow-Prediction_RRL.xlsx` — Review of Related Literature (RRL) synthesis matrix.
- `requirements.txt` — Required R libraries.

## Getting Started

1. Install **R** (and RStudio) along with Python/TensorFlow backend for `keras`/`reticulate` if running deep learning models.
2. Install required R packages:
   ```r
   pkgs <- grep("^#", readLines("requirements.txt"), invert = TRUE, value = TRUE)
   install.packages(pkgs[nzchar(pkgs)])
   ```
3. Point `read.table()` in `TrafficFlow_PESM08DataSet_V2.R` to `./traffic.csv` and execute the analysis.

## License

This project is licensed under the [MIT License](LICENSE).
