import numpy as np
import pandas as pd

np.random.seed(42)
n_samples = 50

years_exp = np.round(np.random.uniform(0.5, 8.0, n_samples), 1)
tech_stack = np.random.randint(1, 7, n_samples)
salary = 10.0 + 4.5 * years_exp + 2.0 * tech_stack + np.random.normal(0, 2.5, n_samples)
salary = np.round(salary, 1)

df = pd.DataFrame({
    'Years_Exp': years_exp,
    'Tech_Stack': tech_stack,
    'Salary_VND': salary
})

print(df.head(), "\n")
print(df.describe(), "\n")

X = df[['Years_Exp', 'Tech_Stack']].values
y = df['Salary_VND'].values

indices = np.arange(n_samples)
np.random.shuffle(indices)
train_size = int(0.8 * n_samples)

train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

theta = np.linalg.inv(X_train_b.T.dot(X_train_b)).dot(X_train_b.T).dot(y_train)

intercept = theta[0]
coef = theta[1:]

print(f"Trọng số: {coef}")
print(f"Độ lệch: {intercept:.4f}")
print(f"Ý nghĩa: Tăng 1 năm kinh nghiệm tăng khoảng {coef[0]:.2f} triệu VNĐ.\n")

y_pred = X_test_b.dot(theta)

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}\n")

sample_dev = np.array([[1.0, 3.5, 4]])
predicted_salary = sample_dev.dot(theta)
print(f"Lương dự đoán: {predicted_salary[0]:.2f} triệu VNĐ\n")

alpha_ridge = 1.0
I = np.eye(X_train_b.shape[1])
I[0, 0] = 0
theta_ridge = np.linalg.inv(X_train_b.T.dot(X_train_b) + alpha_ridge * I).dot(X_train_b.T).dot(y_train)

def train_lasso(X, y, alpha=0.5, lr=0.01, epochs=1000):
    m, n = X.shape
    theta_l = np.zeros(n)
    for _ in range(epochs):
        predictions = X.dot(theta_l)
        errors = predictions - y
        gradients = (1/m) * X.T.dot(errors) + alpha * np.sign(theta_l)
        gradients[0] = (1/m) * X.T[0].dot(errors)
        theta_l -= lr * gradients
    return theta_l

theta_lasso = train_lasso(X_train_b, y_train, alpha=0.5)

comparison_df = pd.DataFrame({
    'Feature': ['Years_Exp', 'Tech_Stack'],
    'LinearRegression': coef,
    'Ridge': theta_ridge[1:],
    'Lasso': theta_lasso[1:]
})

print(comparison_df.to_string(index=False))
print(f"\nIntercepts - Linear: {intercept:.4f} | Ridge: {theta_ridge[0]:.4f} | Lasso: {theta_lasso[0]:.4f}")