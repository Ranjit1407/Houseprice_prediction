import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#load dataset
df = pd.read_csv(r"A:\MLProjects\House\house_prices.csv")

#drop rows which have missing values in the target column
df = df.dropna(subset=["Price (in rupees)"])

#select only numeric columns except the target
numeric_cols = df.select_dtypes(include='number').columns.tolist()
if "Price (in rupees)" in numeric_cols:
    numeric_cols.remove("Price (in rupees)")

#extract features and target
X = df[numeric_cols]
y = df["Price (in rupees)"]

#drop columns that are completely NaN
X = X.dropna(axis=1, how='all')

#fill remaining NaNs with column means
X = X.fillna(X.mean(numeric_only=True))

#final NaN check
if X.isnull().any().any():
    X = X.dropna()
    y = y.loc[X.index]  

corr_matrix = df[numeric_cols + ["Price (in rupees)"]].corr()

#heatmap 
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

#feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#train the model
model = LinearRegression()
model.fit(X_train, y_train)


# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print evaluation
print("✅ Mean Squared Error:", mse)
print("✅ R² Score:", r2)

# Print coefficients
print("\nModel Coefficients:")
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef}")
