import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Simulate some traffic data
np.random.seed(42)

# Number of observations (simulating different times of day, etc.)
num_samples = 1000

# Simulating vehicle counts (car, motorcycle, bus, truck, van)
car_count = np.random.randint(0, 20, num_samples)
motorcycle_count = np.random.randint(0, 10, num_samples)
bus_count = np.random.randint(0, 5, num_samples)
truck_count = np.random.randint(0, 5, num_samples)
van_count = np.random.randint(0, 5, num_samples)

# Calculating the corresponding green light duration based on the vehicle counts and their multipliers
# We assume a basic multiplier for each vehicle type as mentioned above
green_light_duration = (car_count * 1) + (motorcycle_count * 1) + (bus_count * 2) + (truck_count * 2.5) + (van_count * 1.5)

# Create a DataFrame
data = pd.DataFrame({
    'Car': car_count,
    'Motorcycle': motorcycle_count,
    'Bus': bus_count,
    'Truck': truck_count,
    'Van': van_count,
    'Green Light Duration': green_light_duration
})

# Split the data into training and testing sets
X = data[['Car', 'Motorcycle', 'Bus', 'Truck', 'Van']]
y = data['Green Light Duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Calculate the model's performance (R-squared score)
r2_score = model.score(X_test, y_test)
print(f'R-squared score: {r2_score:.2f}')

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Duration')
plt.ylabel('Predicted Duration')
plt.title('Actual vs Predicted Green Light Duration')
plt.show()

# Predict the green light duration for a new set of vehicle counts
new_vehicle_counts = np.array([[5, 2, 1, 1, 1]])  # Example: 5 cars, 2 motorcycles, 1 bus, 1 truck, 1 van
predicted_duration = model.predict(new_vehicle_counts)
print(f'Predicted Green Light Duration for new vehicle counts: {predicted_duration[0]:.2f} seconds')
