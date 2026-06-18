import numpy as np


def truss3d_element_stiffness(x1, x2, E, A):
    """
    计算三维杆单元在全局坐标系下的单元刚度矩阵
    参数:
        x1: 节点1坐标 [x, y, z]
        x2: 节点2坐标 [x, y, z]
        E: 弹性模量 (Pa)
        A: 截面积 (m^2)
    返回:
        L: 单元长度
        direction_cosines: 方向余弦 [cx, cy, cz]
        Ke: 6x6单元刚度矩阵
    """
    # 计算杆长和方向余弦
    dx = x2[0] - x1[0]
    dy = x2[1] - x1[1]
    dz = x2[2] - x1[2]
    L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # 退化单元检查（两点重合）
    if np.isclose(L, 0):
        raise ValueError("错误：两个节点重合，无法形成杆单元！")

    cx = dx / L
    cy = dy / L
    cz = dz / L
    direction_cosines = np.array([cx, cy, cz])

    # 构建方向余弦矩阵T的核心部分（3x3）
    T = np.array([
        [cx, cy, cz, 0, 0, 0],
        [0, 0, 0, cx, cy, cz]
    ])

    # 局部坐标系下的刚度矩阵（1x1）
    k_local = (E * A) / L

    # 全局坐标系下的刚度矩阵 Ke = T^T * k_local * T
    Ke = k_local * (T.T @ T)

    return L, direction_cosines, Ke


def truss3d_element_stress(x1, x2, E, A, de):
    """
    根据节点位移计算单元的轴向应变、应力和轴力
    参数:
        x1: 节点1坐标 [x, y, z]
        x2: 节点2坐标 [x, y, z]
        E: 弹性模量 (Pa)
        A: 截面积 (m^2)
        de: 节点位移向量 [u1, v1, w1, u2, v2, w2] (m)
    返回:
        epsilon: 轴向应变
        sigma: 轴向应力 (Pa)
        N: 轴力 (N)
    """
    # 先获取方向余弦和长度
    L, direction_cosines, _ = truss3d_element_stiffness(x1, x2, E, A)
    cx, cy, cz = direction_cosines

    # 计算轴向应变
    # epsilon = (1/L) * [ -cx, -cy, -cz, cx, cy, cz ] @ de
    B = (1 / L) * np.array([-cx, -cy, -cz, cx, cy, cz])
    epsilon = B @ de

    # 应力和轴力
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


def print_separator():
    print("=" * 60)


# ---------------------- 算例1：沿x轴的一维杆单元 ----------------------
print_separator()
print("【算例1：沿x轴的一维杆单元】")
print_separator()

# 输入参数
x1_1 = np.array([0, 0, 0])
x2_1 = np.array([2, 0, 0])
E1 = 200e9  # 200 GPa
A1 = 1.0e-4  # m^2
de1 = np.array([0, 0, 0, 1.0e-3, 0, 0])  # m

try:
    L1, dc1, Ke1 = truss3d_element_stiffness(x1_1, x2_1, E1, A1)
    epsilon1, sigma1, N1 = truss3d_element_stress(x1_1, x2_1, E1, A1, de1)

    print(f"单元长度 L = {L1} m")
    print(f"方向余弦 cx, cy, cz = {dc1[0]:.4f}, {dc1[1]:.4f}, {dc1[2]:.4f}")
    print("\n单元刚度矩阵 Ke (6x6):")
    print(Ke1)
    print(f"\n轴向应变 epsilon = {epsilon1:.6e}")
    print(f"轴向应力 sigma = {sigma1 / 1e6:.2f} MPa")
    print(f"轴力 N = {N1:.4e} N")
except ValueError as e:
    print(e)

print_separator()

# ---------------------- 算例2：空间任意方向杆单元 ----------------------
print_separator()
print("【算例2：空间任意方向杆单元】")
print_separator()

# 输入参数
x1_2 = np.array([0, 0, 0])
x2_2 = np.array([1, 2, 2])
E2 = 210e9  # 210 GPa
A2 = 2.0e-4  # m^2
de2 = np.array([0, 0, 0, 1.0e-3, 2.0e-3, 2.0e-3])  # m

try:
    L2, dc2, Ke2 = truss3d_element_stiffness(x1_2, x2_2, E2, A2)
    epsilon2, sigma2, N2 = truss3d_element_stress(x1_2, x2_2, E2, A2, de2)

    print(f"单元长度 L = {L2} m")
    print(f"方向余弦 cx, cy, cz = {dc2[0]:.4f}, {dc2[1]:.4f}, {dc2[2]:.4f}")
    print("\n单元刚度矩阵 Ke (6x6):")
    print(Ke2)
    print(f"\n轴向应变 epsilon = {epsilon2:.6e}")
    print(f"轴向应力 sigma = {sigma2 / 1e6:.2f} MPa")
    print(f"轴力 N = {N2:.4e} N")

    # 额外验证：刚度矩阵对称性、刚体位移、特征值
    print("\n--- 刚度矩阵性质验证 ---")
    print(f"刚度矩阵是否对称：{np.allclose(Ke2, Ke2.T)}")

    # 刚体位移验证（平移不产生内力）
    de_rigid = np.array([1, 1, 1, 1, 1, 1])  # 刚体平移
    eps_rigid, sigma_rigid, N_rigid = truss3d_element_stress(x1_2, x2_2, E2, A2, de_rigid)
    print(f"刚体平移时应变：{eps_rigid:.2e}（应为0）")
    print(f"刚体平移时轴力：{N_rigid:.2e} N（应为0）")

    # 特征值验证
    eigvals = np.linalg.eigvalsh(Ke2)
    print(f"刚度矩阵特征值（非负性检查）：\n{eigvals.round(4)}")
    print(f"零特征值个数：{np.sum(np.isclose(eigvals, 0, atol=1e-9))}")
    print("说明：单个杆单元刚度矩阵奇异，因为存在刚体位移模式（单元整体平移/转动），导致矩阵秩亏。")
except ValueError as e:
    print(e)

print_separator()

# ---------------------- 退化单元检查（两点重合） ----------------------
print_separator()
print("【退化单元检查：两点重合测试】")
print_separator()

x1_deg = np.array([0, 0, 0])
x2_deg = np.array([0, 0, 0])
try:
    truss3d_element_stiffness(x1_deg, x2_deg, 200e9, 1e-4)
except ValueError as e:
    print(f"测试结果：{e}（正确提示）")

print_separator()