import cv2
import numpy as np


def rot90(v):
    return np.array([v[1], -v[0]])


def main():
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

    n_ec = 40
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

    qs = np.zeros(n_ec)
    R = np.zeros((2, n_ec, n_ec))  # クーロン力の重み行列(成分ごと)
    for i in range(n_ec):
        for j in range(n_ec):
            pi = ec_poses[i] - 0.5 * ec_normals[i]
            pj = ec_poses[j]
            r = pi - pj
            R[:, i, j] = r / np.linalg.norm(r) ** 3

    for _ in range(100):
        # 電場計算
        E_q = np.dot(R, qs).T  # (n_ec, 2)
        E_const = np.array((0.1, 0))
        E = E_q + E_const

        # 流入する電荷量計算
        q_flow = np.array([np.dot(E[i], ec_normals[i]) for i in range(n_ec)])

        # for i in range(n_ec):
        # print(i, ec_normals[i], q_flow[i])

        # 電荷量の更新
        qs += 0.2 * q_flow

    print(q_flow)

    for i in range(n_ec):
        print(i, ec_normals[i], qs[i])


if __name__ == "__main__":
    main()
