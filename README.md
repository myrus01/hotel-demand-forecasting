# Hotel Demand Forecasting

Forecasting daily guest arrivals for a city hotel one week ahead, so the hotel can plan staffing, purchasing and pricing. The final model cuts the error by 37% compared to the best simple forecast, mainly thanks to features built from hotel-industry logic rather than a more complex algorithm.

## Data

[Hotel booking demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand) — 119,390 bookings for a city hotel and a resort hotel, arrivals from July 2015 to August 2017. About 37% of bookings were cancelled.

This project focuses on the **City Hotel**.

## Building the time series

- Target: number of **actual arrivals** per day (non-cancelled bookings), not all bookings
- Days with zero arrivals were missing after grouping and were filled with 0, so that "7 days ago" always means 7 calendar days
- **July–August 2015 were removed**: arrivals were far below the same months in 2016 and 2017 (August 2015: ~40/day vs ~65–69/day), which points to incomplete data at the start of the records
- Final series: 731 days (September 2015 – August 2017)

## Key patterns

- **Weekly rhythm:** Fridays (~69 arrivals) and Mondays (~64) are busiest, Tuesdays the quietest (~51) — likely leisure and business travellers
- **Yearly seasonality:** winter months drop to ~30–45 arrivals/day, spring to autumn ~65–75
- **Growth:** winter 2016/17 was noticeably stronger than winter 2015/16
- **High daily noise** and occasional spikes (likely group bookings)

## Validation

- **Time-based split:** train on September 2015 – June 2017, test on July–August 2017 (62 days). A random split would let the model see the future
- **Forecast horizon:** 7 days, so every lag and rolling feature is shifted by at least 7 days
- **Metric:** MAE — average error in number of guests per day

## Results

| Approach | MAE |
|---|---|
| Train mean | 14.2 |
| Day-of-week mean | 15.2 |
| Same weekday, average of last 4 weeks | 15.4 |
| Same day last week | 17.6 |
| 28-day rolling mean (best baseline) | 13.9 |
| Gradient boosting: calendar + lags + rolling mean | 14.2 |
| + bookings already on the books | 9.9 |
| **+ expected arrivals adjusted for cancellations** | **8.7** |

### What made the difference

1. **Simple baselines first.** The series is so noisy that copying a single past day performs worst; heavy smoothing of recent data works best.
2. **A more complex model alone didn't help.** Gradient boosting on calendar and lag features (14.2) did not beat the rolling mean.
3. **Domain knowledge did.** Hotels forecast from reservations already on the books. The feature `otb7` counts bookings for each date made at least 7 days in advance, including ones later cancelled, because cancellations are not known yet at forecast time. It became the dominant feature (importance ~0.82) and cut MAE to 9.9.
4. **A cancellation model inside the forecast.** A Random Forest classifier estimates each booking's cancellation probability; summing `1 − p` per date gives expected arrivals. This reduced MAE to 8.7.

### Avoiding data leakage in the cancellation model

Columns that are only known after arrival or cancellation were excluded: `reservation_status`, `reservation_status_date`, `assigned_room_type`, `required_car_parking_spaces`, plus `booking_changes` and `total_of_special_requests` (they change over time) and `country` (often updated at check-in for guests who arrived). The honest ROC-AUC is 0.705; with leaky columns it would look far higher but be useless in practice.

Cancellation probabilities for training-period bookings come from 5-fold cross-validation (`cross_val_predict`), so the forecasting model never sees overconfident in-sample predictions.

## Error analysis

- The model captures the timing of most peaks and drops
- It **underestimates large peaks** (e.g. 115–118 actual vs ~85 predicted): last-minute bookings and group reservations are not visible a week ahead

## Limitations and next steps

- Evaluated on a single 62-day period; backtesting over several periods would make the comparison more reliable
- Add a feature for typical last-minute booking volume and public holidays
- Forecast room-nights (occupancy), not only arrivals

## Setup

1. Download `hotel_bookings.csv` from the Kaggle link above and put it in the project folder
2. Install dependencies:
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
3. Open the notebook in VS Code or Jupyter and run all cells

## What I learned

- Always compare against simple baselines — a complex model is only worth it if it beats them
- Good features beat complex models: one feature from industry logic did more than any algorithm change
- Time series need time-based validation and lags that respect the forecast horizon
- Data leakage can hide in many forms: random splits, post-event columns, in-sample predictions
- Check data quality by comparing a month with the same month in other years, not with its neighbours
.gitignore:
venv/
.ipynb_checkpoints/
*.csv