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
    parser.add_argument('--thickness', type=float, default=1,
                        help='導体板の厚さ [m]')
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
    for ec_pos, ec_normal in zip(ec_poses, ec_normals):

        def to_pt(pos, dx=0, dy=0):
            return (int(pos[0]) + dx, int(pos[1]) + dy)

        cv2.line(img, to_pt(ec_pos, dx=3), to_pt(ec_pos, dx=-3), (0, 0, 255))
        cv2.line(img, to_pt(ec_pos, dy=3), to_pt(ec_pos, dy=-3), (0, 0, 255))

        cv2.line(img, to_pt(ec_pos), to_pt(ec_pos + 10 * ec_normal),
                 (0, 255, 0))

    cv2.imwrite("a.png", img)
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

    print(qs)

    for _ in range(1):
        # 電場計算
        E_q = np.dot(R, qs).T  # (n_ec, 2)
        E = E_q + E_const

        # 流入する電荷量計算
        q_flow = np.array([np.dot(E[i], ec_normals[i]) for i in range(n_ec)])

        print(q_flow)

        # 電荷量の更新
        # qs += args.update_rate * q_flow


if __name__ == "__main__":
    main()
