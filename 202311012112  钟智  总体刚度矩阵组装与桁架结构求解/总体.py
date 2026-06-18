import numpy as np
import json
import os


def read_model_from_json(file_path):
    """前处理：从JSON文件读取桁架模型数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    title = data.get("Title", "Truss Model")
    nsd = data.get("nsd", 2)    # 空间维度
    ndof = data.get("ndof", 2)  # 每个节点自由度
    nnp = data.get("nnp", 0)    # 节点数
    nel = data.get("nel", 0)    # 单元数
    nen = data.get("nen", 2)    # 每个单元节点数

    x = np.array(data.get("x", []), dtype=float)
    y = np.array(data.get("y", []), dtype=float)
    IEN = np.array(data.get("IEN", []), dtype=int) - 1
    E = np.array(data.get("E", []), dtype=float)
    A = np.array(data.get("CArea", []), dtype=float)
    fixed_dof = np.array(data.get("fixed_dof", []), dtype=int) - 1
    fixed_value = np.array(data.get("fixed_value", []), dtype=float)
    force_dof = np.array(data.get("force_dof", []), dtype=int) - 1
    force_value = np.array(data.get("force_value", []), dtype=float)

    return {
        "title": title,
        "nsd": nsd,
        "ndof": ndof,
        "nnp": nnp,
        "nel": nel,
        "nen": nen,
        "x": x,
        "y": y,
        "IEN": IEN,
        "E": E,
        "A": A,
        "fixed_dof": fixed_dof,
        "fixed_value": fixed_value,
        "force_dof": force_dof,
        "force_value": force_value
    }


def compute_element_stiffness(x1, y1, x2, y2, E, A, nsd):
    """单元分析：计算杆单元刚度矩阵（支持1D/2D）"""
    dx = x2 - x1
    dy = y2 - y1
    L = np.sqrt(dx**2 + dy**2)
    if np.isclose(L, 0):
        raise ValueError("单元长度为0，无法形成有效杆单元！")

    if nsd == 1:
        # 一维杆单元（仅x方向）
        Ke = (E * A / L) * np.array([
            [1, -1],
            [-1, 1]
        ])
        return L, 1.0, 0.0, Ke
    elif nsd == 2:
        # 二维杆单元
        c = dx / L
        s = dy / L
        k = E * A / L
        Ke = k * np.array([
            [c**2, c*s, -c**2, -c*s],
            [c*s, s**2, -c*s, -s**2],
            [-c**2, -c*s, c**2, c*s],
            [-c*s, -s**2, c*s, s**2]
        ])
        return L, c, s, Ke
    else:
        raise ValueError("不支持的空间维度！")


def build_LM_matrix(IEN, nnp, ndof):
    """生成对号矩阵LM"""
    nel = IEN.shape[0]
    nen = IEN.shape[1]
    LM = np.zeros((nen * ndof, nel), dtype=int)
    for e in range(nel):
        for i in range(nen):
            node = IEN[e, i]
            for d in range(ndof):
                LM[i * ndof + d, e] = node * ndof + d
    return LM


def assemble_global_stiffness(LM, Ke_list, nnp, ndof):
    """直接组装：根据对号矩阵将单元刚度矩阵组装为总体刚度矩阵"""
    ndof_total = nnp * ndof
    K = np.zeros((ndof_total, ndof_total))
    for e, Ke in enumerate(Ke_list):
        for a in range(Ke.shape[0]):
            for b in range(Ke.shape[1]):
                dof_a = LM[a, e]
                dof_b = LM[b, e]
                K[dof_a, dof_b] += Ke[a, b]
    return K


def apply_boundary_conditions(K, F, fixed_dof, fixed_value):
    """边界条件处理：缩减法求解未知位移"""
    ndof_total = K.shape[0]
    free_dof = np.setdiff1d(np.arange(ndof_total), fixed_dof)
    d_E = fixed_value
    K_FF = K[free_dof, :][:, free_dof]
    F_F = F[free_dof]
    if len(fixed_dof) > 0:
        F_F = F_F - K[free_dof, :][:, fixed_dof] @ d_E
    d_F = np.linalg.solve(K_FF, F_F)
    d = np.zeros(ndof_total)
    d[free_dof] = d_F
    d[fixed_dof] = d_E
    return d, free_dof


def compute_reactions(K, d, fixed_dof):
    """计算约束反力"""
    reactions = K[fixed_dof, :] @ d
    return reactions


def compute_element_stress_and_force(L, c, s, E, d_e, nsd):
    """后处理：计算单元应力和轴力"""
    if nsd == 1:
        epsilon = (1 / L) * np.array([-1, 1]) @ d_e
    elif nsd == 2:
        epsilon = (1 / L) * np.array([-c, -s, c, s]) @ d_e
    else:
        raise ValueError("不支持的空间维度！")
    sigma = E * epsilon
    return epsilon, sigma, sigma * 1  # A在输入中已归一化，直接返回轴力


def solve_truss(file_path):
    """桁架求解主函数"""
    model = read_model_from_json(file_path)
    print(f"=== 模型名称：{model['title']} ===")

    Ke_list = []
    elem_info = []
    for e in range(model['nel']):
        i, j = model['IEN'][e]
        x1, y1 = model['x'][i], model['y'][i]
        x2, y2 = model['x'][j], model['y'][j]
        E = model['E'][e]
        A = model['A'][e]
        L, c, s, Ke = compute_element_stiffness(x1, y1, x2, y2, E, A, model['nsd'])
        Ke_list.append(Ke)
        elem_info.append((L, c, s, E))

    LM = build_LM_matrix(model['IEN'], model['nnp'], model['ndof'])
    K = assemble_global_stiffness(LM, Ke_list, model['nnp'], model['ndof'])
    print("总体刚度矩阵：")
    print(K)
    print("刚度矩阵对称性检查：", np.allclose(K, K.T))

    ndof_total = model['nnp'] * model['ndof']
    F = np.zeros(ndof_total)
    F[model['force_dof']] = model['force_value']
    d, free_dof = apply_boundary_conditions(K, F, model['fixed_dof'], model['fixed_value'])

    print("\n节点位移：")
    for n in range(model['nnp']):
        if model['nsd'] == 1:
            u = d[n * 1]
            print(f"节点{n+1}: u={u:.6f}")
        elif model['nsd'] == 2:
            u = d[n * 2]
            v = d[n * 2 + 1]
            print(f"节点{n+1}: u={u:.6f}, v={v:.6f}")

    print("\n约束反力：")
    reactions = compute_reactions(K, d, model['fixed_dof'])
    for i, dof in enumerate(model['fixed_dof']):
        print(f"自由度{dof+1}反力：{reactions[i]:.6f}")

    print("\n单元应力与轴力：")
    for e in range(model['nel']):
        i, j = model['IEN'][e]
        if model['nsd'] == 1:
            d_e = np.array([d[i * 1], d[j * 1]])
        elif model['nsd'] == 2:
            d_e = np.array([d[i * 2], d[i * 2 + 1], d[j * 2], d[j * 2 + 1]])
        L, c, s, E = elem_info[e]
        eps, sigma, N = compute_element_stress_and_force(L, c, s, E, d_e, model['nsd'])
        print(f"单元{e+1}: 长度={L:.3f}m, 应力={sigma:.6f}, 轴力={N:.6f}")

    return model, K, d, reactions, elem_info


if __name__ == "__main__":
    print("="*60)
    print("【算例1：一维两单元杆结构】")
    print("="*60)
    example1_data = {
        "Title": "1D Two Bar Example",
        "nsd": 1,
        "ndof": 1,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "E": [100, 200],
        "CArea": [1, 1],
        "x": [0, 1, 2],
        "y": [0, 0, 0],
        "IEN": [[1, 2], [2, 3]],
        "fixed_dof": [1],
        "fixed_value": [0.0],
        "force_dof": [3],
        "force_value": [10.0]
    }
    with open("example1.json", "w", encoding="utf-8") as f:
        json.dump(example1_data, f, indent=4)
    model1, K1, d1, reactions1, elem_info1 = solve_truss("example1.json")

    print("\n" + "="*60)
    print("【算例2：二维两杆桁架结构】")
    print("="*60)
    example2_data = {
        "Title": "2D Two Bar Truss Example",
        "nsd": 2,
        "ndof": 2,
        "nnp": 3,
        "nel": 2,
        "nen": 2,
        "E": [1.0, 1.0],
        "CArea": [1.0, 1.0],
        "x": [1.0, 0.0, 1.0],
        "y": [0.0, 0.0, 1.0],
        "IEN": [[1, 3], [2, 3]],
        "fixed_dof": [1, 2, 3, 4],
        "fixed_value": [0.0, 0.0, 0.0, 0.0],
        "force_dof": [5],
        "force_value": [10.0]
    }
    with open("example2.json", "w", encoding="utf-8") as f:
        json.dump(example2_data, f, indent=4)
    model2, K2, d2, reactions2, elem_info2 = solve_truss("example2.json")

