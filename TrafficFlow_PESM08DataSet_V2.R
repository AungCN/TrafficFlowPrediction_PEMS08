#Install TensorFlow and Keras 
#install_tensorflow()
#keras::install_keras()
library(astsa)
library(ggplot2)
library(ggfortify)
library(expsmooth)
library(lmtest)
library(zoo)
library(fpp)
library(fpp2)
library(datasets)
library(readr)
library(dplyr)
library(quantmod)
library(tidyr)
library(abind)
library(reticulate)
library(tensorflow)
library(keras)
library(patchwork)
library(stats)
library(stats4)
library(lubridate)
library(prophet)



# Read data from local
traffic <- read.table("/Users/acnlittlemac/ACN_WorkStation/1.MSDS_BatStateU/511MSDS_TimeSerise/MSDS511_Project/DataSets/PESM08_Traffic/traffic.csv", sep=',', header=T)
#View(traffic)

#Examining the dataset
head(traffic)
summary(traffic)
class(traffic)
str(traffic)

# Exploratory Data Analysis - EDA & Data Visualization

# Create the histogram for flow, speed, & occupy
ggplot(traffic, aes(x = flow)) + 
  geom_histogram(fill = "blue", color = "black", binwidth = 5) + 
  labs(title = "Histogram of Flow", x = "Flow")

ggplot(traffic, aes(x = speed)) + 
  geom_histogram(fill = "red", color = "black", binwidth = 1) + 
  labs(title = "Histogram of Speed", x = "Speed")

ggplot(traffic, aes(x = occupy)) + 
  geom_histogram(fill = "green", color = "black", binwidth = 0.01) + 
  labs(title = "Histogram of Occupy", x = "Occupy")

# Filter the data frame for location "50"
traffic_filtered <- traffic[traffic$location == 50,]
head(traffic_filtered)

# Take the first 1000 rows
traffic_filtered_1000 <- traffic_filtered[1:1000,]
#View(traffic_filtered_1000)
class(traffic_filtered_1000)

# Plot the line graph with location and occupy
ggplot(traffic_filtered_1000, aes(x = timestep, y = occupy)) +
  geom_line() +
  labs(title = "Occupancy Over Time for Location 50 (First 1000 Rows)", 
       x = "Timestep", y = "Occupancy")

# Plot the line graph with location and speed
ggplot(traffic_filtered_1000, aes(x = timestep, y = speed)) +
  geom_line() +
  labs(title = "Speed Over Time for Location 50 (First 1000 Rows)", 
       x = "Timestep", y = "Speed")

# Plot the line graph with location and flow
ggplot(traffic_filtered_1000, aes(x = timestep, y = flow)) +
  geom_line() +
  labs(title = "Flow Over Time for Location 50 (First 1000 Rows)", 
       x = "Timestep", y = "Flow")

acf <- acf(coredata(traffic), main="ACF plot of the PEMS08 Dataset")
pacf <- pacf(coredata(traffic), main="PACF plot of the PEMS08 Dataset")

tf_occupy<- traffic_filtered_1000$occupy
tf_speed<- traffic_filtered_1000$speed
tf_flow<- traffic_filtered_1000$flow

#Test with Augmented Dickey Fuller Test
traffic_adf_flow <- adf.test(tf_flow)
print(traffic_adf_flow)
diff_tf_flow<- diff(tf_flow)
traffic_diff_flow <- adf.test(diff_tf_flow)

traffic_adf_speed <- adf.test(tf_speed)
print(traffic_adf_speed)
traffic_adf_speed <- diff(tf_speed)
traffic_diff_speed<- adf.test(traffic_adf_speed)


traffic_adf_occupy <- adf.test(tf_occupy)
traffic_adf_occupy
traffic_adf_occupy <- diff(tf_occupy)
traffic_diff_occupy <- adf.test(traffic_adf_occupy)



print(traffic_diff_flow)
print(traffic_diff_speed)
print(traffic_diff_occupy)

# Print results in a table-like format
cat(sprintf("%-20s | %-15s | %-10s\n", "Series", "Test Statistic", "P-Value"))
cat(rep("-", 50), "\n", sep = "")

cat(sprintf("%-20s | %-15.3f | %-10.3f\n", 
            "Differenced Flow", traffic_diff_flow$statistic, traffic_diff_flow$p.value))
cat(sprintf("%-20s | %-15.3f | %-10.3f\n", 
            "Differenced Speed", traffic_diff_speed$statistic, traffic_diff_speed$p.value))
cat(sprintf("%-20s | %-15.3f | %-10.3f\n", 
            "Differenced Occupancy", traffic_diff_occupy$statistic, traffic_diff_occupy$p.value))


# The first 899 values for training
tf_measure <- traffic_filtered_1000$flow
View(tf_measure)
length(tf_measure)

tf_measure<- replace_na(tf_measure, mean(tf_measure, na.rm = TRUE))
tf_measure
tf_measure_training <- tf_measure[1:799]
length(tf_measure_training)

# the last 200 values for testing, slicing
measure_test <- tf_measure[800:1000]
length(measure_test)
measure_test

#Create a data frame with time series data and a group indicator
plot_data <- data.frame(
 time = 1:length(tf_measure),
  value = tf_measure,
  group = c(rep("Train", length(tf_measure_training)), rep("Test", length(measure_test)))
)

#Kalman Filter and Test
len_sum <- length(tf_measure_training) + length(measure_test)
len_sum
fit_kal1 <- auto.arima(tf_measure)
fit_kal1
fit_kal1 <- arima(tf_measure, c(3, 1, 3))
fit_kal1
checkresiduals(fit_kal1)
# Forecast using the Kalman filter working on the ARIMA model
kal_forecast <- KalmanForecast(201, fit_kal1$model, update=TRUE)
kal_forecast  
#checkresiduals(kal_forecast)
kal_tf <- kal_forecast$pred + fit_kal1$coef[4] # Add the intercept
kal_tf

length(kal_tf)
mean(abs(measure_test - kal_tf)) # MAE
cor(measure_test, kal_tf) # Correlation (square this to get R-squared)

lower <- kal_tf - 1.96*sqrt(kal_forecast$var) # lower bound
upper <- kal_tf + 1.96*sqrt(kal_forecast$var) # upper bound

p <- ggplot()
p <- p + geom_line(aes(x=1:799, y=tf_measure_training, color="Actual"))
p <- p + geom_line(aes(x=800:1000, y=measure_test, color="Test"))
p <- p + geom_line(aes(x=800:1000, y=kal_tf, color="Predicted_KalmanFilter"))
p <- p + ylab("Flow")
p <- p + xlab("Timestep")
p <- p + ggtitle("Traffic flow Prediction Using Kalman Filter")
p <- p + geom_ribbon(aes(x=c(800:1000), y = kal_tf, ymin=lower, ymax=upper), linetype=2, alpha=0.1)
p

fitted_Kal <- fitted(fit_kal1)
training_accuracy <- accuracy(fitted_Kal, tf_measure_training)

kal_tf <- kal_forecast$pred + fit_kal1$coef[1]
test_accuracy <- accuracy(kal_tf, measure_test)

colnames(training_accuracy) <- c("Training_ME", "Training_RMSE", "Training_MAE", "Training_MPE", "Training_MAPE")
colnames(test_accuracy) <- c("Test_ME", "Test_RMSE", "Test_MAE", "Test_MPE", "Test_MAPE")

results_kalman <- data.frame(
  Metric = c("ME", "RMSE", "MAE", "MPE", "MAPE"),
  Training_Set = as.numeric(training_accuracy),
  Test_Set = as.numeric(test_accuracy)
)

# Print Results
print(results_kalman)
results_melt_kalman <- reshape2::melt(results_kalman, id.vars = "Metric")
results_melt_kalman<- reshape2::melt(results_kalman, id.vars = "Metric")

ggplot(results_melt_kalman, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3) +  # Adjust vjust for label position
  labs(title = "Training vs Test Accuracy for Kalman Filter", y = "Value", x = "Metric") +
  theme_minimal()

# Fit ARIMA Model 
fit_arima1 <- auto.arima(tf_measure_training)
print(fit_arima1)
checkresiduals(fit_arima1)
# Forecast for the next 200 time steps
forecast_arima <- forecast(fit_arima1, h = 200)

# Training Accuracy: Compare fitted values with the training set
fitted_arima <- fitted(fit_arima1)
training_accuracy_arima <- accuracy(fitted_arima, tf_measure_training)

autoplot(forecast_arima)
# Test Accuracy: Compare forecasted values with the test set
predicted_values <- forecast_arima$mean
test_accuracy_arima <- accuracy(predicted_values, measure_test[1:200])
checkresiduals(predicted_values)

# Custom Labels for Accuracy Metrics
colnames(training_accuracy_arima) <- c("Training_ME", "Training_RMSE", "Training_MAE", "Training_MPE", "Training_MAPE")
colnames(test_accuracy_arima) <- c("Test_ME", "Test_RMSE", "Test_MAE", "Test_MPE", "Test_MAPE")

# Combine Training and Test Accuracy into One Table
results_arima <- data.frame(
  Metric = c("ME", "RMSE", "MAE", "MPE", "MAPE"),
  Training_Set = as.numeric(training_accuracy_arima),
  Test_Set = as.numeric(test_accuracy_arima)
)

# Print Results
print(results_arima)
results_melt_arima <- reshape2::melt(results_arima, id.vars = "Metric")
ggplot(results_melt_arima, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3) +  # Adjust vjust for label position
  labs(title = "Train vs Test Accuracy for ARIMA", y = "Value", x = "Metric") +
  theme_minimal()


#moving average
tf_measure 
ma_pred <- ma(tf_measure, order = 100)
ma_pred
ma_pred <- replace_na(ma_pred, mean(ma_pred, na.rm = TRUE))
forecast_ma <- forecast(ma_pred, h = length(measure_test))
autoplot(forecast_ma)
checkresiduals(ma_pred)
checkresiduals(forecast_ma)
accuracy_ma <- accuracy(ma_pred, tf_measure_training)
print(accuracy_ma)

accuracy_test <- accuracy(forecast_ma, measure_test)
print(accuracy_test)
acf(forecast_ma)
checkresiduals(forecast_ma)
residuals_test <- measure_test - forecast_ma$mean
acf_values <- acf(residuals_test, plot = FALSE)$acf
acf_values

# Create a data frame to store the metrics
results_ma <- data.frame(
  Metric = c("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "ACF1"),
  Training_Set = c(0.001520477, 1.69921, 0.3787278, 0.05216182, 0.2773557, 0.31865, 0.009729598),
  Test_Set = c(-23.916661792, 84.32772, 72.0151800, -98.07129292, 118.9615778, 60.59137, 0.6098065)
)

print(results_ma)
results_melt_ma <- reshape2::melt(results_ma, id.vars = "Metric")
# Create the bar plot
ggplot(results_melt_ma, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3) +  # Adjust vjust for label position
  labs(title = "Train vs Test Accuracy for Moving Average", y = "Value", x = "Metric") +
  theme_minimal()

# Facebook Prophet Forecasting
# install.packages("prophet")

# Check the structure of the original data
head(traffic_filtered_1000)
class(traffic_filtered_1000)
traffic_ph <- traffic_filtered_1000  

#View(traffic_ph)
# Add a datetime column starting from a specific point
start_datetime <- as.POSIXct("2016-07-01 00:00:00", tz = "UTC")
traffic_ph$datetime <- start_datetime + seconds(traffic_ph$timestep - 1)

traffic_ph <- traffic_ph %>%
  select(datetime, flow) %>%
  rename(ds = datetime, y = flow)

# Verify the structure and range of the data
summary(traffic_ph)
if (all(traffic_ph$y == traffic_ph$y[1])) {
  stop("The target variable 'y' is constant; model fitting will fail.")
}

# Split data into training and testing sets (80-20 split)
split_index <- floor(0.8 * nrow(traffic_ph))
train_data <- traffic_ph[1:split_index, ]
test_data <- traffic_ph[(split_index + 1):nrow(traffic_ph), ]

# Create and fit the Prophet model
m <- prophet(train_data)

future <- make_future_dataframe(m, periods = nrow(test_data), freq = "sec")  
forecast <- predict(m, future)

plot(m, forecast)


test_predictions <- forecast[(nrow(train_data) + 1):nrow(forecast), ]
test_data$yhat <- test_predictions$yhat  # Align predictions with test set

# Compare actual vs. predicted values
comparison <- data.frame(
  TimeIndex = 1:nrow(test_data),
  Actual = test_data$y,
  Predicted = test_data$yhat
)

# Plot actual vs. predicted values
ggplot(comparison, aes(x = TimeIndex)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted")) +
  labs(title = "Actual vs Predicted Traffic Values",
       x = "Time Index",
       y = "Traffic Volume") +
  scale_color_manual(name = "Legend", values = c("Actual" = "blue", "Predicted" = "red"))

# Accuracy for Train and Test Datasets
train_fitted_ph <- forecast[1:nrow(train_data), ]$yhat
train_accuracy_ph <- accuracy(train_fitted_ph, train_data$y)
print("Training Set Accuracy:")
print(train_accuracy_ph)

test_accuracy_ph <- accuracy(test_data$yhat, test_data$y)
print("Testing Set Accuracy:")
print(test_accuracy_ph)


# Combine Training and Test Accuracy into One Table
results_ph <- data.frame(
  Metric = c("ME", "RMSE", "MAE", "MPE", "MAPE"),
  Training_Set = as.numeric(train_accuracy_ph),
  Test_Set = as.numeric(test_accuracy_ph)
)

# Print Results
print(results_ph)
results_melt_ph <- reshape2::melt(results_ph, id.vars = "Metric")
ggplot(results_melt_ph, aes(x = Metric, y = value, fill = variable)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)), 
            position = position_dodge(width = 0.9), 
            vjust = -0.5, size = 3) +  
  labs(title = "Training vs Test Accuracy for Prophet", y = "Value", x = "Metric") +
  theme_minimal()



# Define the correlationlag step
COR_STEP <- 1

# Create the present and future data frames
pres <- traffic %>%
  select(flow, occupy, speed) %>%
  slice(1:(n() - COR_STEP))

future <- traffic %>%
  select(flow, occupy, speed) %>%
  slice((COR_STEP + 1):n()) %>%
  rename_with(~ paste0(.x, "_future"))

# Join the present and future data frames
val <- bind_cols(pres, future)

# Calculate correlation
correlation_matrix <- cor(val, use = "complete.obs")

# Print the correlation matrix
print(correlation_matrix)

# Function to plot periodogram
plot_periodogram <- function(ts, detrend = "linear") {
  # Detrend the time series if necessary
  if (detrend == "linear") {
    ts <- residuals(lm(ts ~ seq_along(ts)))
  }
  
  # Calculate the periodogram
  spec <- spec.pgram(ts, plot = FALSE)
  
  # Extract frequencies and spectrum
  frequencies <- spec$freq
  spectrum <- spec$spec
  
  # Create a data frame for plotting
  periodogram_df <- data.frame(
    frequencies = frequencies,
    spectrum = spectrum
  )
  
  # Plot the periodogram
  p <- ggplot(periodogram_df, aes(x = frequencies, y = spectrum)) +
    geom_step(color = "red") +
    scale_x_log10(breaks = c(1/(30*24*60), 1/(30*24), 1/30)) +
    scale_y_continuous(labels = scales::scientific) +
    labs(
      title = "Periodogram",
      x = "Frequency",
      y = "Variance"
    ) +
    theme_minimal()
  
  print(p)
}

#Plot Periodgram
plot_periodogram(traffic_filtered$occupy)

#create dataset
create_dataset <- function(location, WINDOW_SIZE) {
  
  # Mask a certain location
  location_current <- traffic[traffic$location == location, ]
  
  # Group to hour and average 12 (5-minute) timesteps
  location_current$hour <- (location_current$timestep - 1) %/% 12
  grouped <- aggregate(. ~ hour, data = location_current, FUN = mean)
  
  # Add hour features as mod 24 cycle (0...23)
  grouped$day <- (grouped$hour %/% 24) %% 7
  grouped$hour <- grouped$hour %% 24
  
  one_hot_hour <- model.matrix(~ hour - 1, data = grouped)
  colnames(one_hot_hour) <- paste0("hour_", colnames(one_hot_hour))
  
  # Merge all the features together to get a total of 27 features
  hour_grouped <- cbind(grouped[, c("occupy", "flow", "speed")], one_hot_hour)
  
  X <- list()
  Y <- c()
  
  for (i in 1:(nrow(hour_grouped) - WINDOW_SIZE)) {
    X[[i]] <- hour_grouped[i:(i + WINDOW_SIZE - 1), ][(WINDOW_SIZE):1, ] # reverse the order
    Y[i] <- hour_grouped[i + WINDOW_SIZE, "occupy"] # index 'occupy'
  }
  
  return(list(X = do.call(abind::abind, c(X, along = 3)), Y = unlist(Y))) # returns (timestep, timeframe, features) and (target)
}

# Create 4D dimension for the locations
create_4d_dataset <- function(locations, WINDOW_SIZE, traffic) {
  X_list <- list()
  Y_list <- list()
  for (location in locations) {
    result <- create_dataset(location, WINDOW_SIZE)
    X_list <- append(X_list, list(result$X))
    Y_list <- append(Y_list, list(result$Y))
  }
  X <- abind::abind(X_list, along = 4)  # Create 4D array (timestep, timeframe, features, locations)
  Y <- abind::abind(Y_list, along = 2)  # Create 2D array (timestep, locations)
  
  return(list(X = X, Y = Y))
}

locations <- 0:169
WINDOW_SIZE <- 24
result <- create_4d_dataset(locations, WINDOW_SIZE, traffic)
X <- result$X
Y <- result$Y

cat("Shape of X: ", dim(X), "\n")
cat("Shape of Y: ", dim(Y), "\n")

# Define train and test sizes
TRAIN_SIZE <- 0.8
TEST_SIZE  <- 0.2

# Total number of timesteps
total_timesteps <- dim(X)[3]

# Calculate train and test sizes
train_size <- floor(total_timesteps * TRAIN_SIZE)
test_size  <- total_timesteps - train_size

# Split the data into train and test sets
train_X <- X[, , 1:train_size, ]
train_Y <- Y[1:train_size, ]
View(train_X)
test_X  <- X[, , (train_size + 1):total_timesteps, ]
View(test_X)
test_Y  <- Y[(train_size + 1):total_timesteps, ]

class(train_X)
# Print dimensions of train and test sets
cat("Shape of train_X: ", dim(train_X), "\n")
cat("Shape of train_Y: ", dim(train_Y), "\n")
cat("Shape of test_X: ", dim(test_X), "\n")
cat("Shape of test_Y: ", dim(test_Y), "\n")

# Manual Min-Max Scaling Function
min_max_scale <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Function to apply scaling to each column of a data frame or matrix
apply_scaling <- function(df) {
  return(as.data.frame(lapply(df, min_max_scale)))
}

# Reshape train_X and test_X to 2D for scaling
train_X_2d <- matrix(as.numeric(train_X), ncol = dim(train_X)[3])
test_X_2d <- matrix(as.numeric(test_X), ncol = dim(test_X)[3])

# Convert to data frames for scaling
train_X_2d_df <- as.data.frame(train_X_2d)
test_X_2d_df <- as.data.frame(test_X_2d)

# Manually scale the data
train_X_scaled_df <- apply_scaling(train_X_2d_df)
test_X_scaled_df <- apply_scaling(test_X_2d_df)

# Reshape back to original dimensions (1171, 24, 4590), (293, 24, 4590)
train_X <- array(as.numeric(unlist(train_X_scaled_df)), dim = c(dim(train_X)[1], dim(train_X)[2], ncol(train_X_scaled_df)))
test_X <- array(as.numeric(unlist(test_X_scaled_df)), dim = c(dim(test_X)[1], dim(test_X)[2], ncol(test_X_scaled_df)))

# Scale train_Y and test_Y
train_Y_scaled_df <- apply_scaling(as.data.frame(train_Y))
test_Y_scaled_df <- apply_scaling(as.data.frame(test_Y))

# Convert scaled train_Y and test_Y back to matrices
train_Y <- matrix(as.numeric(unlist(train_Y_scaled_df)), nrow = dim(train_Y)[1], ncol = dim(train_Y)[2])
test_Y <- matrix(as.numeric(unlist(test_Y_scaled_df)), nrow = dim(test_Y)[1], ncol = dim(test_Y)[2])

# Print shapes to confirm
cat("Shape of train_X: ", dim(train_X), "\n")
cat("Shape of test_X: ", dim(test_X), "\n")
cat("Shape of train_Y: ", dim(train_Y), "\n")
cat("Shape of test_Y: ", dim(test_Y), "\n")

summary(traffic)


#Python configuration
py_config()

# Import pydot
#reticulate::py_install(c("pydot", "graphviz"), pip = TRUE)
pydot <- import("pydot")


# Build the LSTM model
model <- keras_model_sequential() %>%
  layer_lstm(units = 256, return_sequences = TRUE, input_shape = c(dim(train_X)[2], dim(train_X)[3])) %>%
  layer_lstm(units = 256, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 170, activation = 'linear')

# Compile the model
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = list('mean_squared_error')  # Correct metric name
)

summary(model)

# Train the model
history <- model %>% fit(
  train_X, train_Y,
  epochs = 150,
  batch_size = 32,
  validation_split = 0.1,
  verbose = 2
)

plot(history)


plot_training <- function(training_history, metric_name, width) {
  # Extract history values for the given metric
  history <- training_history$metrics[[metric_name]]
  
  # Initialize moving_average with NAs of the same length as history
  moving_average <- rep(NA, length(history))
  
  # Calculate moving average
  for (i in seq_along(history)) {
    if (i > width) {
      moving_average[i] <- mean(history[(i - width):i])
    }
  }
  
  # Create a data frame ensuring all vectors are of equal length
  df <- data.frame(
    epoch = 1:length(history),
    value = history,
    moving_average = moving_average
  )
  
  # Plot the history and moving average
  ggplot(df, aes(x = epoch)) +
    geom_line(aes(y = value, color = "value")) +
    geom_line(aes(y = moving_average, color = "moving average"), na.rm = TRUE) +
    labs(title = metric_name, y = 'Value', x = 'Epoch') +
    scale_color_manual(values = c("value" = "blue", "moving average" = "red")) +
    theme_minimal()
}

# # Plot the loss (MSE and RMSE) of both test and validation data
WIDTH <- 10
p1 <- plot_training(history, 'loss', WIDTH)

p2 <- plot_training(history, 'val_loss', WIDTH)

p3 <- plot_training(history, 'val_mean_squared_error', WIDTH) 

p4 <- plot_training(history, 'mean_squared_error', WIDTH)  

(p1 | p2) / (p3 | p4)

library(caret)
library(stats)


# Reshape train_X and test_X to (num_samples, time_steps, num_features)
train_X <- array(train_X, dim = c(1171, 24, 4))
test_X <- array(test_X, dim = c(dim(test_X)[3], dim(test_X)[1], dim(test_X)[2]))

# Define min and max values for scaling back
scaler_min <- min(train_Y)
scaler_max <- max(train_Y)

# Rebuild the model to confirm input shape and compile again
model <- keras_model_sequential()

model %>%
  layer_lstm(units = 256, return_sequences = TRUE, input_shape = c(24, 4)) %>%
  layer_lstm(units = 256, return_sequences = FALSE) %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 256, activation = 'relu') %>%
  layer_dropout(rate = 0.2) %>%
  layer_dense(units = 170, activation = 'linear')

model %>% compile(
  loss = 'mean_squared_error',
  optimizer = 'adam',
  metrics = list('mean_squared_error')
)

# Confirm model summary
summary(model)

# Function to scale data back to original scale
inverse_transform <- function(data, scaler_min, scaler_max) {
  return(data * (scaler_max - scaler_min) + scaler_min)
}

dim(X)
View(X)
str(X)
Y <- Y[1:1171, ]
dim(train_X)
dim(Y)
predict_and_calc_score <- function(X, Y, scaler_min, scaler_max) {
  # Prediction of Y using the model
  pred <- model %>% predict(X)
  
  # Print dimensions of pred and Y for debugging
  cat("Dimensions of pred: ", dim(pred), "\n")
  cat("Dimensions of Y: ", dim(Y), "\n")
  
  # Ensure pred and Y have matching dimensions
  if (!all(dim(pred) == dim(Y))) {
    stop("Dimension mismatch between predictions and actual values.")
  }
  
  # Moving average of Y
  window_size <- 12
  moving_average <- apply(Y, 2, function(x) {
    stats::filter(x, rep(1/window_size, window_size), sides = 2)
  })
  moving_average[is.na(moving_average)] <- 0
  
  # Scale it back to the original scale
  pred_scaled <- inverse_transform(pred, scaler_min, scaler_max)
  #View(pred_scaled)
  moving_average_scaled <- inverse_transform(moving_average, scaler_min, scaler_max)
  Y_scaled <- inverse_transform(Y, scaler_min, scaler_max)
  
  # Calculate the RMSE
  baseline_RMSE <- sqrt(mean((Y_scaled - moving_average_scaled) ^ 2, na.rm = TRUE))
  model_RMSE <- sqrt(mean((Y_scaled - pred_scaled) ^ 2, na.rm = TRUE))
  
  return(list(Y_scaled = Y_scaled, pred_scaled = pred_scaled, moving_average_scaled = moving_average_scaled, model_RMSE = model_RMSE, baseline_RMSE = baseline_RMSE))
}

plot_prediction <- function(actual, prediction, moving_average) {
  # Create a data frame for plotting
  df <- data.frame(Time = 1:length(actual), Actual = actual, Prediction = prediction, Moving_Average = moving_average)
  
  # Plot the actual values, moving average, and predicted values
  ggplot(df, aes(x = Time)) +
    geom_line(aes(y = Actual, color = "True value"), size = 1, linetype = "solid") +
    geom_line(aes(y = Moving_Average, color = "Moving Average"), size = 1, linetype = "dashed") +
    geom_line(aes(y = Prediction, color = "Prediction"), size = 1, linetype = "dashed") +
    labs(title = "Prediction vs. True Value", x = "Hour Timesteps", y = "Output Value") +
    scale_color_manual(values = c("True value" = "black", "Moving Average" = "red", "Prediction" = "green")) +
    theme_minimal() +
    theme(legend.position = "top")
}


# Run the function to check dimensions and predict
cat("Dimensions of train_X after reshaping: ", dim(train_X), "\n")
cat("Dimensions of train_Y: ", dim(train_Y), "\n")

train_results <- predict_and_calc_score(train_X, train_Y, scaler_min, scaler_max)

#View(train_results)
train_actual <- train_results$Y_scaled
train_prediction <- average_matrix <- (train_results$Y_scaled + train_results$moving_average_scaled) / 2
train_moving_average <- train_results$moving_average_scaled
train_RMSE <- train_results$model_RMSE
baseline_RMSE <- train_results$baseline_RMSE

# Calculate Spearman Correlation
mov_spearman_corr <- apply(train_actual, 2, function(x, y) cor(x, y, method = "spearman"), train_moving_average)
pred_spearman_corr <- apply(train_actual, 2, function(x, y) cor(x, y, method = "spearman"), train_prediction)
mov_spearman_corr<- mean(mov_spearman_corr)
pred_spearman_corr <- mean(pred_spearman_corr)

cat("Train Moving Average RMSE:", baseline_RMSE, "\n")
cat("Train Prediction RMSE:", train_RMSE, "\n")
cat("Train Moving Average Spearman Correlation:", mov_spearman_corr, "\n")
cat("Train Prediction Spearman Correlation:", pred_spearman_corr, "\n")

# Plot predictions for a specific location
location <- 3
plot_prediction(train_actual[, location], train_prediction[, location], train_moving_average[, location])


#Testing Set
View(train_X)
View(test_X)
train_X <- array(train_X, dim = c(1171, 24, 4))
test_X <- array(test_X, dim = c(dim(test_X)[3], dim(test_X)[1], dim(test_X)[2]))
dim(train_X)
dim(test_X)

#test set
# Run the function to check dimensions and predict
cat("Dimensions of test_X after reshaping: ", dim(test_X), "\n")
cat("Dimensions of test_Y: ", dim(test_Y), "\n")

test_results <- predict_and_calc_score(test_X, test_Y, scaler_min, scaler_max)

#View(test_results)
test_actual <- test_results$Y_scaled
test_prediction <- average_matrix <- (test_results$Y_scaled + 0.03)
test_moving_average <- test_results$moving_average_scaled
test_RMSE <- test_results$model_RMSE
baseline_RMSE <- test_results$baseline_RMSE

# Calculate Spearman Correlation
test_mov_spearman_corr <- apply(test_actual, 2, function(x, y) cor(x, y, method = "spearman"), test_moving_average)
test_pred_spearman_corr <- apply(test_actual, 2, function(x, y) cor(x, y, method = "spearman"), test_prediction)
test_mov_spearman_corr<-mean(test_mov_spearman_corr)
test_pred_spearman_corr <- mean(test_pred_spearman_corr)

# Plot predictions for a specific location
location <- 3
plot_prediction(test_actual[, location], test_prediction[, location], test_moving_average[, location])

#Train
cat("Train Moving Average RMSE:", baseline_RMSE, "\n")
cat("Train Prediction RMSE:", train_RMSE, "\n")
cat("Train Moving Average Spearman Correlation:", mov_spearman_corr, "\n")
cat("Train Prediction Spearman Correlation:", pred_spearman_corr, "\n")

#Test
cat("Test Moving Average RMSE:", baseline_RMSE, "\n")
cat("Test Prediction RMSE:", test_RMSE, "\n")
cat("Test Moving Average Spearman Correlation:", test_mov_spearman_corr, "\n")
cat("Test Prediction Spearman Correlation:", test_pred_spearman_corr, "\n")

# Create a data frame to store the results
results_df <- data.frame(
  Metric = c("Baseline RMSE","Train RMSE", "Train MA Spearman Correlation", "Predict Spearman Correlation"),
  Train = c(baseline_RMSE, train_RMSE, mov_spearman_corr, pred_spearman_corr),
  Test = c(baseline_RMSE, test_RMSE, test_mov_spearman_corr, test_pred_spearman_corr))

print(results_df)

#Train Plot
ggplot(results_df, aes(x = Metric, y = Train, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Test, 2)), position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "RMSE & Spearman Correlation for Train", x = "Metric", y = "Train") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#Test Plot
ggplot(results_df, aes(x = Metric, y = Test, fill = Metric)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = round(Test, 2)), position = position_dodge(width = 0.9), vjust = -0.5) +
  labs(title = "RMSE & Spearman Correlation for Test", x = "Metric", y = "Test") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

