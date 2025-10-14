# 输出角度改为 [0, 360) 区间的版本（仍保持算法与验证相同）
from math import sin, cos, asin, atan2, radians, degrees
import numpy as np
import random

def wrap_deg_360(a):
    """把角度 wrap 到 [0, 360)"""
    return a % 360.0

def wrap_triplet_deg_360(tri):
    az, el, roll = tri
    return (wrap_deg_360(az), wrap_deg_360(el), wrap_deg_360(roll))

def triad_from_aer_deg(az_deg, el_deg, roll_deg):
    """(az, el, roll) -> 三基向量 {f, r, u}（单位化）"""
    az = radians(az_deg); el = radians(el_deg); rho = radians(roll_deg)
    ca, sa = np.cos(az), np.sin(az)
    cb, sb = np.cos(el), np.sin(el)
    cr, sr = np.cos(rho), np.sin(rho)
    f = np.array([cb*ca, cb*sa, sb], dtype=float)
    u0 = np.array([-sb*ca, -sb*sa, cb], dtype=float)
    r0 = np.array([-sa,     ca,    0], dtype=float)
    u = u0*cr + r0*sr
    r = r0*cr - u0*sr
    # 数值稳健的单位化
    f /= np.linalg.norm(f); r /= np.linalg.norm(r); u /= np.linalg.norm(u)
    return f, r, u

def az_el_from_dir_deg_360(d):
    """单位向量 d -> (az, el)（角度，最后 wrap 到 [0,360)）"""
    x, y, z = d
    az = degrees(np.arctan2(y, x)) % 360.0
    # el 由 arcsin，原始范围 [-90, 90]；按用户需求也 wrap 到 [0, 360)
    el = degrees(np.arcsin(np.clip(z, -1.0, 1.0))) % 360.0
    return az, el

def default_up_from_dir(d):
    """给定 forward d，roll=0 定义下默认上方向 w0(d)"""
    x, y, z = d
    az = np.arctan2(y, x)
    el = np.arcsin(np.clip(z, -1.0, 1.0))
    return np.array([
        -np.sin(el)*np.cos(az),
        -np.sin(el)*np.sin(az),
         np.cos(el)
    ], dtype=float)

def roll_from_up(d, desired_up):
    """
    已知 forward=d 与目标上方向 desired_up，求 roll，使得“上方向==desired_up”
    与本文 triad 约定一致： roll = atan2( d·(w × w0), w0·w )
    """
    w0 = default_up_from_dir(d)
    w = desired_up / np.linalg.norm(desired_up)
    num = float(np.dot(d, np.cross(w, w0)))   # 使用 w × w0
    den = float(np.dot(w0, w))
    return (degrees(np.arctan2(num, den))) % 360.0

def views_from_front_360(az_deg, el_deg, roll_deg):
    """
    输出角度均 wrap 到 [0, 360)
    - right：看向正视图 r，且其“上”与正视图 u 对齐
    - up   ：看向正视图 u，且其“上”与 -f 对齐
    """
    f, r, u = triad_from_aer_deg(az_deg, el_deg, roll_deg)
    # right
    az_r, el_r = az_el_from_dir_deg_360(r)
    roll_r = roll_from_up(r, u)
    # up
    az_u, el_u = az_el_from_dir_deg_360(u)
    roll_u = roll_from_up(u, -f)
    dL = -r
    az_l, el_l = az_el_from_dir_deg_360(dL)
    roll_l = roll_from_up(dL, u)
    dD = -u
    az_d, el_d = az_el_from_dir_deg_360(dD)
    roll_d = roll_from_up(dD, f)
    return {
        "front": wrap_triplet_deg_360((az_deg, el_deg, roll_deg)),
        "right": wrap_triplet_deg_360((az_r,  el_r,  roll_r)),
        "up":    wrap_triplet_deg_360((az_u,  el_u,  roll_u)),
        #"down":  wrap_triplet_deg_360((az_d,  el_d,  roll_d)),
        #'left': wrap_triplet_deg_360((az_l, el_l, roll_l))
    }
    return {
        "front": wrap_triplet_deg_360((az_deg, el_deg, roll_deg)),
        "right": wrap_triplet_deg_360((az_r,  el_r,  roll_r)),
        "up":    wrap_triplet_deg_360((az_u,  el_u,  roll_u))
    }

def left_from_front_360(az_deg, el_deg, roll_deg):
    """左视图（输出三元素都在 [0, 360)）：看向 -r，且“上”与正视图 u 对齐"""
    f, r, u = triad_from_aer_deg(az_deg, el_deg, roll_deg)
    dL = -r
    az_l, el_l = az_el_from_dir_deg_360(dL)
    roll_l = roll_from_up(dL, u)
    return wrap_triplet_deg_360((az_l, el_l, roll_l))

def triads_close(a, b, tol=1e-9):
    """比较两组三基 {f,r,u} 是否等价（逐向量最大绝对差 < tol）"""
    fa, ra, ua = triad_from_aer_deg(*a)
    fb, rb, ub = triad_from_aer_deg(*b)
    return (np.max(np.abs(fa-fb)) < tol and
            np.max(np.abs(ra-rb)) < tol and
            np.max(np.abs(ua-ub)) < tol)

# ---------------- 验证 ----------------
if __name__ == "__main__":
    # 一般化 round-trip：left 作为 front 的 right 要回到原 front
    random.seed(42)
    for i in range(5):
        az = random.uniform(0, 360)
        el = random.uniform(0, 360)  # 允许 0-360 表示
        roll = random.uniform(0, 360)
        V = views_from_front_360(az, el, roll)
        left = left_from_front_360(az, el, roll)
        V_from_left = views_from_front_360(*left)
        # 由于算法内部使用三向量比较，这里把角度直接送入，不需要再额外 wrap
        assert triads_close(V["front"], V_from_left["right"]), "round-trip failed"

    # 特例：roll=0 时退化到题给公式（但输出被 wrap 到 [0,360)）
    random.seed(7)
    for i in range(5):
        az = random.uniform(0, 360)
        el = random.uniform(0, 360)
        V = views_from_front_360(az, el, 0.0)
        right_expected = wrap_triplet_deg_360((az + 90.0, 0.0, el))
        up_expected    = wrap_triplet_deg_360((az, el + 90.0, 0.0))
        assert triads_close(V["right"], right_expected)
        assert triads_close(V["up"],    up_expected)

    print("✅ 已切换到 [0, 360) 区间；反向验证与特例验证均通过。")
