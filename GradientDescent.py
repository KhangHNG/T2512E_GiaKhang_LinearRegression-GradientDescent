import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 100

X = np.random.uniform(1.0, 10.0, n_samples)
y_real = 5.0 + 2.5 * X + np.random.normal(0, 1.5, n_samples)

print(f"Kích thước tập X: {X.shape}, Tập y: {y_real.shape}")
print(f"5 mẫu đầu tiên - Chi phí X: {X[:5].round(2)}")
print(f"5 mẫu đầu tiên - Doanh số y: {y_real[:5].round(2)}\n")

# Task 1: Viết hàm tính MSE Loss và Đạo hàm Gradients
def compute_loss(X, y, w, b):
    n = len(y)
    y_pred = w * X + b
    loss = (1 / n) * np.sum((y_pred - y) ** 2)
    return loss

def compute_gradients(X, y, w, b):
    n = len(y)
    y_pred = w * X + b
    dw = (2 / n) * np.dot((y_pred - y), X)
    db = (2 / n) * np.sum(y_pred - y)
    return dw, db

# Task 2: Xây dựng vòng lặp Gradient Descent
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000

history_loss = []

for epoch in range(1, epochs + 1):
    loss = compute_loss(X, y_real, w, b)
    history_loss.append(loss)
    
    dw, db = compute_gradients(X, y_real, w, b)
    
    w -= learning_rate * dw
    b -= learning_rate * db
    
    if epoch in [1, 200, 400, 600, 800, 1000]:
        print(f"Epoch {epoch:4d} | w: {w:.4f} | b: {b:.4f} | MSE Loss: {loss:.4f}")

print(f"\nKết quả cuối cùng sau {epochs} epochs -> w: {w:.4f}, b: {b:.4f}\n")

# Task 3: Trực quan hóa tiến trình hội tụ
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(epochs), history_loss, color='blue', linewidth=2)
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(True)

axes[1].scatter(X, y_real, color='black', alpha=0.7, label='Dữ liệu thực tế')
X_line = np.linspace(X.min(), X.max(), 100)
y_line = w * X_line + b
axes[1].plot(X_line, y_line, color='red', linewidth=2, label=f'Đường hồi quy: y = {w:.2f}X + {b:.2f}')
axes[1].set_title("Hồi quy tuyến tính với Gradient Descent")
axes[1].set_xlabel("Chi phí quảng cáo (X)")
axes[1].set_ylabel("Doanh số (y)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Task 4: Thử nghiệm Tỷ lệ học (Learning Rate Experiment)
def run_gradient_descent(X, y, lr, epochs=200):
    w_t, b_t = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        losses.append(compute_loss(X, y, w_t, b_t))
        dw, db = compute_gradients(X, y, w_t, b_t)
        w_t -= lr * dw
        b_t -= lr * db
    return losses, w_t, b_t

lr_cases = [0.0001, 0.01, 0.15]
plt.figure(figsize=(8, 5))

for lr in lr_cases:
    losses, final_w, final_b = run_gradient_descent(X, y_real, lr, epochs=200)
    plt.plot(losses, label=f'lr = {lr}')
    print(f"Learning Rate {lr} -> Cuối epoch 200 Loss: {losses[-1]:.4f} (w: {final_w:.4f}, b: {final_b:.4f})")

plt.title("So sánh các mức Learning Rate (200 Epochs)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()