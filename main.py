import argparse
import cv2
import numpy as np


def rot90(v):
    return np.array([v[1], -v[0]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ec', type=int, default=40,
                        help='電荷の配置数')
    parser.add_argument('--pixel_len', type=float, default=1,
                        help='1ピクセルが表す長さ [m]')
    parser.add_argument('--eps0', type=float, default=1,
                        help='真空の誘電率 [Fm-1]')
    args = parser.parse_args()

    img = cv2.imread("circle.png")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    bin_img = 255 - bin_img

    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
    )
    """
    print(cnt[0].shape)
    cv2.drawContours(img, cnt, -1, color=(0, 0, 255), thickness=1)

    cv2.imwrite("a.png", img)
    """

    # 電荷の位置とそこでの法線を計算する
    total_len = 0
    for contour in contours:
        n = len(contour)
        total_len += sum([np.linalg.norm(contour[i] - contour[i - 1])
                          for i in range(n)])

    n_ec = args.n_ec
    ec_poses = []
    ec_normals = []
    for i in range(n_ec):
        # NOTE: 二乗オーダーになっているが、計算時間的に別に問題ないはず
        rest = i * total_len / n_ec
        for contour in contours:
            n = len(contour)
            for i in range(n):
                len_i = np.linalg.norm(contour[i] - contour[i - 1])
                if rest <= len_i:
                    ratio = rest / len_i
                    pos = (1 - ratio) * contour[i - 1, 0] +\
                        ratio * contour[i, 0]
                    vec = contour[i - 1, 0] - contour[i, 0]
                    normal = rot90(vec / np.linalg.norm(vec))
                    ec_poses.append(pos)
                    ec_normals.append(normal)
                    break
                else:
                    rest -= len_i
            else:
                continue
            break
    ec_poses = np.array(ec_poses)
    ec_normals = np.array(ec_normals)

    """
    qs = np.zeros(n_ec)  # 該当領域の総電荷
    A = args.thickness * total_len / n_ec  # 該当領域の面積
    R = np.zeros((2, n_ec, n_ec))  # クーロン力の重み行列(成分ごと)
    for i in range(n_ec):
        for j in range(n_ec):
            if i != j:
                # 十分離れている場合は単に点電荷とみなして計算する
                pi = ec_poses[i]
                pj = ec_poses[j]
                r = (pi - pj) * args.pixel_len
                coef = 1 / (4 * np.pi * args.eps0)
                R[:, i, j] = coef * (r / np.linalg.norm(r) ** 3)
            else:
                # 自身の表面の電場を計算する場合は面密度に応じて計算する
                R[:, i, j] = -ec_normals[i] / (2 * args.eps0 * A)

    # 線形方程式にして解く
    # ec_normals[:, 0] = 3
    # ec_normals[:, 1] = 1

    M = (R[0].T * ec_normals[:, 0] + R[1].T * ec_normals[:, 1]).T
    E_const = np.array([0.1, 0])
    b = -np.dot(E_const, ec_normals.T)

    qs = np.linalg.solve(M, b)
    """

    # ポテンシャルベースで電荷量を求める
    E_const = np.array([0.1, 0])
    M = np.zeros((n_ec, n_ec))
    a = total_len / n_ec / 2
    inner_poses = []
    for i in range(n_ec):
        inner_poses.append(ec_poses[i] + 4 * a * ec_normals[i])

    for i in range(n_ec):
        for j in range(n_ec):
            """
            pi = inner_poses[i]
            pj = ec_poses[j]
            r = np.linalg.norm(pi - pj) * args.pixel_len
            M[i, j] = 1 / (4 * np.pi * args.eps0 * r)
            """
            if i != j:
                pi = ec_poses[i]
                pj = ec_poses[j]
                r = np.linalg.norm(pi - pj) * args.pixel_len
                M[i, j] = 1 / (4 * np.pi * args.eps0 * r)
            else:
                M[i, j] = 1 / (2 * np.pi * args.eps0 * a)

    # b = np.array([np.dot(E_const, inner_poses[i]) for i in range(n_ec)])
    b = np.array([np.dot(E_const, ec_poses[i]) for i in range(n_ec)])
    qs = np.linalg.solve(M, -b)

    print(np.dot(M, qs) + b)
    return

    A = 4 * a ** 2

    """
    # 電場計算
    E_q = np.dot(R, qs).T  # (n_ec, 2)
    E = E_q + E_const

    # 流入する電荷量計算
    q_flow = np.array([np.dot(E[i], ec_normals[i]) for i in range(n_ec)])

    print(q_flow)
    """

    # 電荷密度
    q_densities = qs / A

    print(q_densities)

    # ポテンシャル計算
    height, width, _ = img.shape
    nx = 20
    ny = 20
    pot_poses = []
    pots = []
    for ix in range(nx):
        for iy in range(ny):
            x = (ix + 0.5) * width / nx
            y = (iy + 0.5) * height / ny
            p = np.array([x, y])

            pot = np.dot(E_const, p)
            for i in range(n_ec):
                r = np.linalg.norm(p - ec_poses[i]) * args.pixel_len
                pot += qs[i] / (4 * np.pi * args.eps0 * max(a / 2, r))

            pot_poses.append(p)
            pots.append(pot)

    q_lim = np.max(np.abs(q_densities))
    pot_lim = np.max(np.abs(pots))

    def to_pt(pos, dx=0, dy=0):
        return (int(pos[0]) + dx, int(pos[1]) + dy)

    for ec_pos, q_density in zip(ec_poses, q_densities):
        intensity = int(191 * abs(q_density) / q_lim) + 64
        if q_density < 0:
            color = (intensity, 64, 64)
        else:
            color = (64, 64, intensity)

        cv2.line(img, to_pt(ec_pos, dx=3), to_pt(ec_pos, dx=-3), color)
        cv2.line(img, to_pt(ec_pos, dy=3), to_pt(ec_pos, dy=-3), color)

    for pos, pot in zip(pot_poses, pots):
        v = abs(pot) * 8 / pot_lim + 1
        cv2.circle(img, to_pt(pos), int(v), (0, 255, 0))

    cv2.imwrite("a.png", img)

    return

    # 電場計算
    height, width, _ = img.shape
    nx = 1
    ny = 1
    ef_poses = []
    ef_vec = []
    for ix in range(nx):
        for iy in range(ny):
            x = (ix + 0.5) * width / nx
            y = (iy + 0.5) * height / ny
            p = np.array([x, y])

            Ep = np.zeros(2)
            Ep = E_const.copy()
            for i in range(n_ec):
                r = (p - ec_poses[i]) * args.pixel_len
                coef = 1 / (4 * np.pi * args.eps0)
                Ep += coef * qs[i] * (r / np.linalg.norm(r) ** 3)

            ef_poses.append(p)
            ef_vec.append(Ep)

    print(ef_vec)

    q_lim = np.max(np.abs(q_densities))
    vec_lim = np.median(np.abs(ef_vec))

    def to_pt(pos, dx=0, dy=0):
        return (int(pos[0]) + dx, int(pos[1]) + dy)

    for ec_pos, q_density in zip(ec_poses, q_densities):
        intensity = int(191 * abs(q_density) / q_lim) + 64
        if q_density < 0:
            color = (intensity, 64, 64)
        else:
            color = (64, 64, intensity)

        cv2.line(img, to_pt(ec_pos, dx=3), to_pt(ec_pos, dx=-3), color)
        cv2.line(img, to_pt(ec_pos, dy=3), to_pt(ec_pos, dy=-3), color)

    for pos, vec in zip(ef_poses, ef_vec):
        v = vec * 8 / vec_lim
        if np.linalg.norm(v) > 8:
            v *= 8 / np.linalg.norm(v)
        cv2.line(img, to_pt(pos), to_pt(pos + v), (0, 255, 0))
        cv2.circle(img, to_pt(pos), 2, (0, 255, 0))

    cv2.imwrite("a.png", img)


if __name__ == "__main__":
    main()
