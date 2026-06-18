import numpy as np
import matplotlib.pyplot as plt
from math import cosh, sinh, exp, tanh

# 全局字体修复：解决中文、希腊字母方框乱码
plt.rcParams["font.family"] = ["SimHei", "DejaVu Sans", "Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "cm"

# ===================== 函数1：计算SUPG最优alpha =====================
def alpha_supg(Pe):
    """返回SUPG最优稳定参数 alpha_opt = coth(Pe) - 1/Pe"""
    if abs(Pe) < 1e-12:
        return 0.0
    return 1.0 / tanh(Pe) - 1.0 / Pe

# ===================== 函数2：生成单元刚度矩阵 =====================
def element_matrix(kappa, v, le, alpha):
    """
    两节点线性单元对流扩散单元矩阵Ke (2×2)
    kappa_bar = kappa + alpha * v * le / 2
    Ke = kappa_bar/le * [[1, -1], [-1, 1]] + v/2 * [[-1, 1], [-1, 1]]
    """
    kappa_bar = kappa + alpha * v * le / 2.0
    K_diff = (kappa_bar / le) * np.array([[1, -1], [-1, 1]])
    K_conv = (v / 2.0) * np.array([[-1, 1], [-1, 1]])
    Ke = K_diff + K_conv
    return Ke

# ===================== 函数3：总体组装+求解 =====================
def solve_advection_diffusion(nel, L, v, kappa, alpha):
    """
    输入：单元数nel，区间长度L，对流速度v，扩散系数kappa，稳定参数alpha
    输出：节点坐标x、数值解theta_num、精确解theta_exact、总刚矩阵K_global
    """
    le = L / nel
    nnodes = nel + 1
    x = np.linspace(0, L, nnodes)

    K_global = np.zeros((nnodes, nnodes))
    F_global = np.zeros(nnodes)

    for e in range(nel):
        i_local = e
        j_local = e + 1
        Ke = element_matrix(kappa, v, le, alpha)
        K_global[i_local, i_local] += Ke[0, 0]
        K_global[i_local, j_local] += Ke[0, 1]
        K_global[j_local, i_local] += Ke[1, 0]
        K_global[j_local, j_local] += Ke[1, 1]

    # 边界条件
    K_global[0, :] = 0.0
    K_global[0, 0] = 1.0
    F_global[0] = 0.0

    K_global[-1, :] = 0.0
    K_global[-1, -1] = 1.0
    F_global[-1] = 1.0

    theta_num = np.linalg.solve(K_global, F_global)

    # 精确解
    theta_exact = np.zeros_like(x)
    Pe_global = v * L / kappa
    denom = exp(Pe_global) - 1.0
    for idx, xi in enumerate(x):
        theta_exact[idx] = (exp(v * xi / kappa) - 1.0) / denom

    return x, theta_num, theta_exact, K_global

# ===================== 函数4：批量计算三种格式 =====================
def compute_all_schemes(nel, L, v, Pe_target):
    le = L / nel
    kappa = v * le / (2.0 * Pe_target)

    x_gal, theta_gal, theta_exact, K_gal = solve_advection_diffusion(nel, L, v, kappa, alpha=0.0)
    err_gal = np.max(np.abs(theta_gal - theta_exact))

    x_upw, theta_upw, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=1.0)
    err_upw = np.max(np.abs(theta_upw - theta_exact))

    alpha_opt = alpha_supg(Pe_target)
    x_supg, theta_supg, _, _ = solve_advection_diffusion(nel, L, v, kappa, alpha=alpha_opt)
    err_supg = np.max(np.abs(theta_supg - theta_exact))

    result = {
        "x": x_gal,
        "exact": theta_exact,
        "galerkin": theta_gal,
        "upwind": theta_upw,
        "supg": theta_supg,
        "err_gal": err_gal,
        "err_upw": err_upw,
        "err_supg": err_supg,
        "K_galerkin": K_gal,
        "kappa": kappa,
        "Pe": Pe_target
    }
    return result

# ===================== 绘图函数【彻底修复el变量报错】 =====================
def plot_result(res, Pe):
    plt.figure(figsize=(10, 6))
    plt.plot(res["x"], res["exact"], 'k-', linewidth=2, label='精确解')
    plt.plot(res["x"], res["galerkin"], 'r--', linewidth=1.5, label=r'标准Galerkin $(\alpha=0)$')
    plt.plot(res["x"], res["upwind"], 'g-.', linewidth=1.5, label=r'迎风格式 $(\alpha=1)$')
    plt.plot(res["x"], res["supg"], 'b:', linewidth=1.5, label=r'SUPG/Petrov-Galerkin $(\alpha_{opt})$')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\theta(x)$')
    # 修复：拆分字符串，r原始字符串单独处理数学公式，不会解析{el}
    plt.title(f'单元Peclet数 $Pe = {Pe}$，单元数量 ' + r'$n_{el}=20$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ===================== 主程序入口 =====================
if __name__ == "__main__":
    L = 1.0
    nel = 20
    v = 1.0
    Pe_list = [0.1, 3.0]

    print("========== 一维对流扩散有限元作业计算结果 ==========\n")
    error_table = []

    for Pe in Pe_list:
        print(f"------------ 当前单元Pe = {Pe} ------------")
        res = compute_all_schemes(nel, L, v, Pe)
        plot_result(res, Pe)
        eg = res["err_gal"]
        eu = res["err_upw"]
        es = res["err_supg"]
        print(f"标准Galerkin最大节点误差: {eg:.6e}")
        print(f"迎风格式最大节点误差:     {eu:.6e}")
        print(f"SUPG最大节点误差:         {es:.6e}")
        error_table.append([Pe, eg, eu, es])

        if abs(Pe - 3.0) < 1e-6:
            K = res["K_galerkin"]
            print("\n--- Pe=3.0 标准Galerkin总体矩阵分析 ---")
            is_sym = np.allclose(K, K.T, atol=1e-10)
            print(f"矩阵是否对称：{is_sym}")
            eigvals = np.linalg.eigvals(K)
            is_pos_def = np.all(eigvals > -1e-10)
            print(f"矩阵是否正定：{is_pos_def}")
            print(f"最小特征值：{np.min(eigvals):.4e}")
            print("前5×5总刚矩阵：")
            print(K[:5, :5])

    print("\n========== 误差汇总表 ==========")
    print(f"{'Pe':<6}{'Galerkin误差':<15}{'迎风格误差':<15}{'SUPG误差':<15}")
    for row in error_table:
        pe, eg, eu, es = row
        print(f"{pe:<6.1f}{eg:<15.6e}{eu:<15.6e}{es:<15.6e}")