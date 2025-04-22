# Classify-Vehicles-Based-on-Engine-Emissions

# Vehicle Emissions Classification

This project builds a machine learning model to classify vehicles into emission categories (A, B, C) based on engine size, CO2 emissions, and fuel type. It uses a Random Forest Classifier for prediction and demonstrates preprocessing techniques including encoding, scaling, and data splitting.

## 📊 Dataset

The dataset contains information about various vehicles including:

- `engine_size`: Size of the engine (in liters)
- `fuel_type`: Type of fuel used (`petrol`, `diesel`, or `electric`)
- `co2_emissions`: CO2 emissions in g/km
- `emission_category`: Emission class label (A, B, C)

## 🧠 Model Pipeline

1. **Load and inspect data**
2. **Label encode** categorical features (`fuel_type` and `emission_category`)
3. **Feature selection**: `engine_size`, `co2_emissions`, and encoded `fuel_type`
4. **Train-test split** (80/20)
5. **Feature scaling** using `StandardScaler`
6. **Model training** using `RandomForestClassifier`
7. **Model evaluation**: accuracy score and classification report
8. **Prediction** on a sample vehicle

## 🔍 Example Prediction

You can predict the emission category of a sample vehicle with:

```python
sample_vehicle = pd.DataFrame([[2.0, 215.41, le_fuel.transform(['petrol'])[0]]], 
                              columns=['engine_size', 'co2_emissions', 'Fuel_Type_Encoded'])
sample_scaled = scaler.transform(sample_vehicle)
predicted = model.predict(sample_scaled)
print(le_target.inverse_transform(predicted))
