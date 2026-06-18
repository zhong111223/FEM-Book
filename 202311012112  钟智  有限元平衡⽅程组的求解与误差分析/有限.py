import numpy as np
import json
import time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import norm
from numpy.linalg import cond as sp_cond

# ====================== 全局配置 ======================
INDEX_BASE = 0  # 数组下标从0开始
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ====================== 任务1：自研LDLT稠密求解器 ======================
def ldlt_factor(K: np.ndarray):
    """
    LDL^T分解 K = L @ D @ L.T
    :param K: n×n 对称方阵
    :return: L(单位下三角), D(对角向量), flag(True正定/False非正定)
    """
    n = K.shape[0]
    L = np.eye(n, dtype=np.float64)
    D = np.zeros(n, dtype=np.float64)
    # 遍历每一行主元
    for j in range(n):
        # 计算D[j]前半部分
        sum_d = K[j, j]
        for k in range(j):
            sum_d -= L[j, k] ** 2 * D[k]
        D[j] = sum_d
        # 检测非正主元
        if D[j] <= 1e-12:
            print(f"【分解错误】第{j}个主元 D[{j}] = {D[j]:.2e} ≤ 0，矩阵非正定/存在零主元")
            return None, None, False
        # 计算L[j+1:, j]
        for i in range(j + 1, n):
            sum_L = K[i, j]
            for k in range(j):
                sum_L -= L[i, k] * L[j, k] * D[k]
            L[i, j] = sum_L / D[j]
    return L, D, True

def ldlt_solve(L: np.ndarray, D: np.ndarray, R: np.ndarray):
    """
    前代 → 对角求解 → 回代 求解 L D L^T a = R
    """
    n = len(D)
    # 1. 前代 L y = R
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = R[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s
    # 2. 对角 D z = y
    z = np.zeros(n, dtype=np.float64)
    for i in range(n):
        z[i] = y[i] / D[i]
    # 3. 回代 L^T a = z
    a = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = z[i]
        for j in range(i + 1, n):
            s -= L[j, i] * a[j]
        a[i] = s
    return a

def residual_norm(K: np.ndarray, a: np.ndarray, R: np.ndarray):
    """计算残差r=R-Ka，残差2范数，相对残差"""
    r = R - K @ a
    norm_r = norm(r, 2)
    norm_R = norm(R, 2)
    rel_r = norm_r / norm_R if norm_R > 1e-15 else 0.0
    return r, norm_r, rel_r

# ====================== 误差与条件数工具 ======================
def calc_error(a_num: np.ndarray, a_exact: np.ndarray):
    """绝对误差范数、相对误差"""
    diff = a_num - a_exact
    norm_diff = norm(diff, 2)
    norm_exact = norm(a_exact, 2)
    rel_err = norm_diff / norm_exact if norm_exact > 1e-15 else 0.0
    return norm_diff, rel_err

def round_to_sig(x, sig=4):
    """四舍五入保留sig位有效数字"""
    if np.abs(x) < 1e-15:
        return 0.0
    scale = 10 ** (sig - np.floor(np.log10(np.abs(x))) - 1)
    return np.round(x * scale) / scale

def mat_round_sig(mat, sig=4):
    vec_func = np.vectorize(lambda x: round_to_sig(x, sig))
    return vec_func(mat)

# ====================== 统一求解接口 solve_equilibrium ======================
def solve_equilibrium(K_FF: np.ndarray, rhs: np.ndarray, method="ldlt", **options):
    """
    缩减平衡方程 K_FF d_F = rhs 统一求解接口
    method: ldlt / sparse_csr
    """
    n = K_FF.shape[0]
    t0 = time.perf_counter()
    if method == "ldlt":
        L, D, flag = ldlt_factor(K_FF)
        if not flag:
            return None, {"success": False, "time": time.perf_counter() - t0}
        d_F = ldlt_solve(L, D, rhs)
        t_solve = time.perf_counter() - t0
        r, nr, rr = residual_norm(K_FF, d_F, rhs)
        info = {
            "success": True,
            "L": L, "D": D,
            "residual_norm": nr,
            "rel_residual": rr,
            "solve_time": t_solve,
            "cond": sp_cond(K_FF)
        }
        return d_F, info
    elif method == "sparse_csr":
        K_sp = csr_matrix(K_FF)
        d_F = spsolve(K_sp, rhs)
        t_solve = time.perf_counter() - t0
        r, nr, rr = residual_norm(K_FF, d_F, rhs)
        info = {
            "success": True,
            "sparse_nnz": K_sp.nnz,
            "residual_norm": nr,
            "rel_residual": rr,
            "solve_time": t_solve,
            "solver_name": "scipy-sparse-CSR(等效PARDISO)"
        }
        return d_F, info
    else:
        raise NotImplementedError(f"不支持求解方法: {method}")

# ====================== 2.3桁架作业复用模块（一维杆/二维桁架） ======================
class TrussFE23:
    """复用2.3作业桁架组装、自由度分块、后处理"""
    def __init__(self):
        self.nodes = None
        self.elems = None
        self.mat_E = None
        self.mat_A = None
        self.LM = None
        self.K_full = None
        self.F_full = None
        self.free_dof = None
        self.known_dof = None
        self.d_known = None
        self.K_FF = None
        self.K_EF = None
        self.rhs = None

    def case1_1d_2bar(self):
        """2.3算例1：一维两单元杆"""
        self.K_full = np.array([
            [100, -100, 0],
            [-100, 300, -200],
            [0, -200, 200]
        ], dtype=np.float64)
        self.F_full = np.array([0, 0, 10], dtype=np.float64)
        # 边界：d0=0（已知位移），d1,d2自由
        self.known_dof = [0]
        self.d_known = np.array([0.0])
        self.free_dof = [1, 2]
        # 分块K_FF, K_EF
        nf = len(self.free_dof)
        ne = len(self.known_dof)
        K_FF = np.zeros((nf, nf))
        K_EF = np.zeros((ne, nf))
        for i, fi in enumerate(self.free_dof):
            for j, fj in enumerate(self.free_dof):
                K_FF[i, j] = self.K_full[fi, fj]
            for e, ei in enumerate(self.known_dof):
                K_EF[e, i] = self.K_full[ei, fi]
        self.K_FF = K_FF
        self.K_EF = K_EF
        # rhs = f_F - K_EF^T @ d_E
        f_F = self.F_full[self.free_dof]
        self.rhs = f_F - self.K_EF.T @ self.d_known
        return

    def post_1d(self, d_F):
        """一维杆后处理：完整位移、约束反力、单元轴力"""
        n_full = self.K_full.shape[0]
        d_full = np.zeros(n_full)
        d_full[self.known_dof] = self.d_known
        d_full[self.free_dof] = d_F
        # 约束反力 R_E = K_EF @ d_F + K_EE @ d_E - F_E
        K_EE = self.K_full[np.ix_(self.known_dof, self.known_dof)]
        F_E = self.F_full[self.known_dof]
        R_E = self.K_EF @ d_F + K_EE @ self.d_known - F_E
        # 单元轴力（两杆单元）
        # 单元1：节点0-1，刚度100
        N1 = 100 * (d_full[1] - d_full[0])
        # 单元2：节点1-2，刚度200
        N2 = 200 * (d_full[2] - d_full[1])
        return {
            "full_displacement": d_full,
            "support_reaction": R_E,
            "elem_force": [N1, N2]
        }

# ====================== Poisson方程有限元 Q4四边形单元 ======================
def poisson_q4_assemble(nx, ny):
    """单位正方形Q4单元，制造解u=sinπx sinπy，返回K_FF, rhs, 坐标映射"""
    L = 1.0
    dx = L / nx
    dy = L / ny
    # 节点坐标 (nx+1)*(ny+1)
    nnx = nx + 1
    nny = ny + 1
    xx = np.linspace(0, L, nnx)
    yy = np.linspace(0, L, nny)
    node_coords = []
    node_id = np.zeros((nnx, nny), dtype=int)
    idx = 0
    for j in range(nny):
        for i in range(nnx):
            node_coords.append([xx[i], yy[j]])
            node_id[i, j] = idx
            idx += 1
    total_nodes = idx
    # Dirichlet边界：x=0,x=1,y=0,y=1全部固定
    known_dof = []
    free_dof = []
    for j in range(nny):
        for i in range(nnx):
            x, y = node_coords[node_id[i, j]]
            if abs(x) < 1e-12 or abs(x - 1) < 1e-12 or abs(y) < 1e-12 or abs(y - 1) < 1e-12:
                known_dof.append(node_id[i, j])
            else:
                free_dof.append(node_id[i, j])
    n_free = len(free_dof)
    free_map = {dof: i for i, dof in enumerate(free_dof)}
    # Q4单元刚度与载荷
    def q4_elem_stiff(x0, y0, hx, hy):
        Ke = np.zeros((4, 4))
        Fe = np.zeros(4)
        pi2 = np.pi ** 2
        f_val = lambda x, y: 2 * pi2 * np.sin(np.pi * x) * np.sin(np.pi * y)
        # 两点高斯积分
        gpt = [-1 / np.sqrt(3), 1 / np.sqrt(3)]
        gw = [1.0, 1.0]
        detJ = hx * hy / 4
        for xi in gpt:
            for eta in gpt:
                dN_dxi = np.array([-(1 - eta)/4, (1 - eta)/4, (1 + eta)/4, -(1 + eta)/4])
                dN_deta = np.array([-(1 - xi)/4, -(1 + xi)/4, (1 + xi)/4, (1 - xi)/4])
                dNdx = dN_dxi * 2 / hx
                dNdy = dN_deta * 2 / hy
                B = np.vstack([dNdx, dNdy])
                Ke += (B.T @ B) * detJ
                # 形函数
                N = np.array([
                    (1 - xi)*(1 - eta)/4,
                    (1 + xi)*(1 - eta)/4,
                    (1 + xi)*(1 + eta)/4,
                    (1 - xi)*(1 + eta)/4
                ])
                xg = x0 + hx * (xi + 1) / 2
                yg = y0 + hy * (eta + 1) / 2
                Fe += N * f_val(xg, yg) * detJ
        return Ke, Fe
    # COO稀疏存储
    rows = []
    cols = []
    vals = []
    rhs_vec = np.zeros(n_free)
    # 遍历单元
    for ey in range(ny):
        for ex in range(nx):
            x0 = ex * dx
            y0 = ey * dy
            # 四个节点
            n1 = node_id[ex, ey]
            n2 = node_id[ex+1, ey]
            n3 = node_id[ex+1, ey+1]
            n4 = node_id[ex, ey+1]
            elem_dofs = [n1, n2, n3, n4]
            Ke, Fe = q4_elem_stiff(x0, y0, dx, dy)
            # 组装刚度
            for a in range(4):
                da = elem_dofs[a]
                if da not in free_map:
                    continue
                ia = free_map[da]
                rhs_vec[ia] += Fe[a]
                for b in range(4):
                    db = elem_dofs[b]
                    if db not in free_map:
                        continue
                    ib = free_map[db]
                    rows.append(ia)
                    cols.append(ib)
                    vals.append(Ke[a, b])
    K_coo = coo_matrix((vals, (rows, cols)), shape=(n_free, n_free))
    K_FF = K_coo.tocsr()
    # 理论解映射
    u_exact_all = np.zeros(total_nodes)
    for i in range(total_nodes):
        x, y = node_coords[i]
        u_exact_all[i] = np.sin(np.pi * x) * np.sin(np.pi * y)
    u_exact_free = np.array([u_exact_all[d] for d in free_dof])
    meta = {
        "nx": nx, "ny": ny,
        "total_nodes": total_nodes,
        "n_elem": nx * ny,
        "n_free": n_free,
        "free_dof": free_dof,
        "node_coords": node_coords,
        "u_exact_all": u_exact_all,
        "u_exact_free": u_exact_free
    }
    return K_FF, rhs_vec, meta

def plot_poisson_result(meta, u_num_free):
    """绘制数值解云图、误差云图"""
    nnx = meta["nx"] + 1
    nny = meta["ny"] + 1
    u_num_all = meta["u_exact_all"].copy()
    for i, dof in enumerate(meta["free_dof"]):
        u_num_all[dof] = u_num_free[i]
    x = np.array([p[0] for p in meta["node_coords"]]).reshape(nnx, nny)
    y = np.array([p[1] for p in meta["node_coords"]]).reshape(nnx, nny)
    u_num = u_num_all.reshape(nnx, nny)
    u_ex = meta["u_exact_all"].reshape(nnx, nny)
    err = np.abs(u_num - u_ex)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cf1 = ax1.contourf(x, y, u_num, cmap="jet")
    plt.colorbar(cf1, ax=ax1)
    ax1.set_title("Poisson方程数值解云图")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    cf2 = ax2.contourf(x, y, err, cmap="coolwarm")
    plt.colorbar(cf2, ax=ax2)
    ax2.set_title("节点绝对误差云图")
    ax2.set_xlabel("x")
    plt.tight_layout()
    plt.show()
    # 计算误差指标
    max_err = np.max(err)
    l2_diff = norm(u_num_free - meta["u_exact_free"], 2)
    l2_ex = norm(meta["u_exact_free"], 2)
    l2_rel = l2_diff / l2_ex
    return max_err, l2_rel

# ====================== 算例执行函数 ======================
def run_case0_truss_1d():
    """算例0：2.3一维两杆桁架"""
    print("=" * 60)
    print("【算例0 一维两杆桁架 2.3作业接口验证】")
    truss = TrussFE23()
    truss.case1_1d_2bar()
    d_F, info = solve_equilibrium(truss.K_FF, truss.rhs, method="ldlt")
    print(f"求解成功：{info['success']}")
    print(f"自由位移 d2, d3 = {d_F[0]:.4f}, {d_F[1]:.4f}")
    post = truss.post_1d(d_F)
    print(f"完整节点位移：{post['full_displacement']}")
    print(f"节点1约束反力：{post['support_reaction'][0]:.4f}")
    print(f"单元1轴力：{post['elem_force'][0]:.4f}, 单元2轴力：{post['elem_force'][1]:.4f}")
    print(f"相对残差：{info['rel_residual']:.2e}, 条件数：{info['cond']:.2e}")
    print("=" * 60 + "\n")

def run_case1_3diag(n_list=[10, 100, 500]):
    """算例1 三对角对称正定矩阵，性能计时"""
    print("【算例1 三对角矩阵规模测试】")
    for n in n_list:
        K = np.zeros((n, n))
        for i in range(n):
            K[i, i] = 2.0
            if i > 0:
                K[i, i-1] = -1.0
                K[i-1, i] = -1.0
        a_exact = np.ones(n)
        R = K @ a_exact
        d_num, info = solve_equilibrium(K, R, method="ldlt")
        _, rel_err = calc_error(d_num, a_exact)
        print(f"n={n:4d} | 求解时间={info['solve_time']:.4e}s | 相对误差={rel_err:.2e}")
    print("\n")

def run_case2_non_posdef():
    """算例2 非正定矩阵检测"""
    print("【算例2 非正定矩阵检测】")
    K = np.array([[1, 2], [2, 1]])
    R = np.array([1, 1])
    d, info = solve_equilibrium(K, R, method="ldlt")
    print(f"分解成功？{info['success']}\n")

def run_case3_ill_condition():
    """任务2 病态矩阵残差与误差分析（修复：增加分解判断，避免None传入求解）"""
    print("【任务2 病态矩阵残差与误差分析】")
    K = np.array([[1.0, 1.0], [1.0, 1.0001]])
    a_exact = np.array([1.0, 1.0])
    R = K @ a_exact
    cond_K = sp_cond(K)
    print(f"矩阵条件数 cond(K) = {cond_K:.2e}")

    # 1. 双精度完整矩阵计算
    L, D, flag = ldlt_factor(K)
    if flag:
        a_double = ldlt_solve(L, D, R)
        r_d, nr_d, rr_d = residual_norm(K, a_double, R)
        _, err_d = calc_error(a_double, a_exact)
        print("=== 双精度计算 ===")
        print(f"解：{a_double}, 相对残差={rr_d:.2e}, 相对误差={err_d:.2e}")
    else:
        print("双精度矩阵分解失败！")

    # 2. 4位有效数字截断矩阵（增加判断，分解失败则跳过求解）
    K_4 = mat_round_sig(K, 4)
    R_4 = mat_round_sig(R, 4)
    L4, D4, flag4 = ldlt_factor(K_4)
    print("=== 4位有效数字截断 ===")
    if flag4:
        a_4 = ldlt_solve(L4, D4, R_4)
        r_4, nr_4, rr_4 = residual_norm(K, a_4, R)
        _, err_4 = calc_error(a_4, a_exact)
        print(f"解：{a_4}, 相对残差={rr_4:.2e}, 相对误差={err_4:.2e}")
    else:
        print("截断4位有效数字后矩阵奇异，无法分解求解，跳过计算")
    print("\n")

def run_case4_poisson(nx=50, ny=50):
    """算例4 Poisson Q4有限元稀疏求解"""
    print(f"【算例4 Poisson Q4有限元 nx={nx}, ny={ny} 稀疏求解】")
    t_assemble0 = time.perf_counter()
    K_sp, rhs, meta = poisson_q4_assemble(nx, ny)
    t_assemble = time.perf_counter() - t_assemble0
    # 稀疏求解
    t_solve0 = time.perf_counter()
    u_num_free = spsolve(K_sp, rhs)
    t_solve = time.perf_counter() - t_solve0
    total_t = t_assemble + t_solve
    # 误差
    max_err, l2_rel = plot_poisson_result(meta, u_num_free)
    # 残差
    rr = norm(K_sp @ u_num_free - rhs, 2) / norm(rhs, 2)
    print(f"单元：Q4四边形 | 总节点：{meta['total_nodes']} | 自由度数：{meta['n_free']}")
    print(f"矩阵非零元：{K_sp.nnz} | 装配时间={t_assemble:.4e}s | 求解时间={t_solve:.4e}s | 总时间={total_t:.4e}s")
    print(f"相对残差={rr:.2e} | 最大节点误差={max_err:.2e} | L2相对误差={l2_rel:.2e}\n")

def export_json_input():
    """输出JSON算例输入文件"""
    test_mat = {
        "Title": "LDLT test",
        "n": 3,
        "K": [[100, -100, 0], [-100, 300, -200], [0, -200, 200]],
        "R": [0, 0, 10]
    }
    with open("input_ldlt_test.json", "w", encoding="utf-8") as f:
        json.dump(test_mat, f, indent=2, ensure_ascii=False)
    print("已生成输入文件 input_ldlt_test.json\n")

# ====================== 主程序入口 ======================
if __name__ == "__main__":
    export_json_input()
    run_case0_truss_1d()
    run_case1_3diag(n_list=[10, 100, 500])
    run_case2_non_posdef()
    run_case3_ill_condition()
    run_case4_poisson(nx=50, ny=50)