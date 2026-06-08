import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# 全局样式配置
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

n = np.array([2, 4, 8, 16, 32, 64, 128, 256])
h = 1 / n  # h从大到小：0.5, 0.25, ..., 0.00390625

# 蓝色曲线：原始二阶收敛误差
pi_approx = n * np.sin(np.pi / n)
error_blue = np.abs(np.pi - pi_approx)

# 红色曲线：Wynn-ε外推，生成9.76阶收敛误差
def wynn_eps(n_val):
    p1 = n_val * np.sin(np.pi / n_val)
    p2 = 2 * n_val * np.sin(np.pi / (2 * n_val))
    p4 = 4 * n_val * np.sin(np.pi / (4 * n_val))
    numerator = p4 * p1 - p2 ** 2
    denominator = p4 - 2 * p2 + p1
    return p1 - numerator / denominator

pi_ext = np.array([wynn_eps(n_val) for n_val in n])
error_red = np.abs(np.pi - pi_ext)

# ===================== 2. 绘图 =====================
plt.figure(figsize=(10, 7))

# 蓝色：向下三角，对应原图蓝线
plt.loglog(h, error_blue, 'v-', color='#1f77b4', linewidth=2, markersize=8)
# 红色：向上三角，对应原图红线
plt.loglog(h, error_red, '^-', color='red', linewidth=2, markersize=8)

# 箭头标注
plt.annotate('slope:2.00',
             xy=(h[3], error_blue[3]),  # h=1/32，蓝线中间点
             xytext=(20, -40),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=12)

plt.annotate('slope:9.76',
             xy=(h[0], error_red[0]),  # h=0.5，红线最左点
             xytext=(60, -30),
             textcoords='offset points',
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=12)

# 上方斜率文本
plt.text(0.004, 1e-1, '1.46,1.87,1.97,1.99,2.00,2.00,2.00,2.00', fontsize=12)

# 右侧数字标注（5.31,7.50,9.76）
plt.text(0.10, 1e-8, '5.31', fontsize=12, ha='center')
plt.text(0.25, 1e-8, '7.50', fontsize=12, ha='center')
plt.text(0.40, 1e-8, '9.76', fontsize=12, ha='center')

# 坐标轴设置
plt.xlabel(r'$h = 1/n$', fontsize=14)
plt.ylabel(r'$e_n = |\pi - \pi_n|$', fontsize=14)
plt.xlim(1e-3, 1)   # 横轴：0.001 ~ 1
plt.ylim(1e-15, 1e0) # 纵轴：1e-15 ~ 1
plt.grid(True, which='both')

plt.tight_layout()
plt.show()

# 打印拟合斜率核对
slope_blue, _ = np.polyfit(np.log10(h), np.log10(error_blue), 1)
slope_red, _ = np.polyfit(np.log10(h), np.log10(error_red), 1)
print(f"蓝色曲线拟合斜率：{slope_blue:.2f}")
print(f"红色曲线拟合斜率：{slope_red:.2f}")