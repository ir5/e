import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def create_sphere_surface_mesh(n_lat, n_lon):
    """
    球面をメッシュによって近似し、三角化を施した結果を返す

    返り値は (メッシュ数, 3角形頂点, xyz) の ndarray
    """
    u = np.linspace(0, 2 * np.pi, n_lat + 1)
    v = np.linspace(-np.pi / 2, np.pi / 2, n_lon + 1)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    r = np.cos(v)
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = np.sin(v)

    tri = Delaunay(np.vstack([u, v]).T)

    p = np.vstack([x, y, z]).T

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    q = p[tri.simplices]
    # ax.scatter(q[:, 0], q[:, 1], q[:, 2], c=q[:, 2])
    # fig.savefig("a.png")

    # 面積0の縮退したメッシュは除く
    q = np.array([mat for mat in q if not np.isclose(np.linalg.det(mat), 0, atol=1e-6)])

    return q


def sample_triangle_points(mat, n_samples):
    """
    三角形内の点をランダムサンプリングする
    """
    u = np.random.uniform(0, 1, size=n_samples)
    v = np.random.uniform(0, 1, size=n_samples)
    over = u + v >= 1
    u[over] = 1 - u[over]
    v[over] = 1 - v[over]

    # 縦ベクトル化しておく
    u = u.reshape(-1, 1)
    v = v.reshape(-1, 1)

    return mat[0] + u * (mat[1] - mat[0]) + v * (mat[2] - mat[0])


def area(mat):
    """
    三角形の面積
    """
    a = mat[1] - mat[0]
    b = mat[2] - mat[0]
    return np.sqrt(np.dot(a, a) * np.dot(b, b) - np.dot(a, b) ** 2) / 2


def approx_surface_potential(mat, eps0=1/(4*np.pi), n_iter=1000):
    """
    xyz空間中の三角形に一様に電荷が分布しているとき、その重心のポテンシャルをモンテカルロ計算する。

    ここでは三角形内の総電荷量は単位量と仮定する。
    """
    ps = sample_triangle_points(mat, n_iter)
    gs = np.mean(mat, axis=0)

    # print(gs)
    # print(np.linalg.norm(ps - gs, axis=1))

    integ_approx = np.mean(1 / np.linalg.norm(ps - gs, axis=1))
    # 面積の項は積分近似と電荷密度の項で打ち消し合うのでここでは現れないことに注意
    return integ_approx / (4 * np.pi * eps0)


def get_pot_of_const_ef(e0):
    """
    一様な静電場におけるポテンシャル関数を返す
    """
    def pot(p):
        return np.dot(p, e0)
    return pot


def solve_core(mesh, pot0, eps0=1/(4*np.pi)):
    """
    ポテンシャル条件を満たす荷量を計算する
    """

    gs = mesh.mean(axis=1)
    n = mesh.shape[0]

    # 表面ポテンシャルを計算する
    pot_surface = np.array([approx_surface_potential(mat) for mat in mesh])

    # M[i, j] := 位置 j に単位量の電荷があるときの位置 i のポテンシャル
    def distmat2(v):
        xx, yy = np.meshgrid(v, v)
        return (xx - yy) ** 2

    d = np.sqrt(distmat2(gs[:, 0]) + distmat2(gs[:, 1]) + distmat2(gs[:, 2]))
    print(d)
    for i in range(n):
        d[i, i] = 1e-18  # 警告が鬱陶しいので非ゼロにしておく
    M = 1 / (4 * np.pi * eps0 * d)
    for i in range(n):
        M[i, i] = pot_surface[i]

    # TODO: 本当は非対角成分で対角より大きい値があったら潰すべきだが、あまりなさそうなので飛ばす
    qs = np.linalg.solve(M, pot0)
    As = np.array([area(mat) for mat in mesh])
    print(np.sum(qs))
    return qs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_ec', type=int, default=40,
                        help='電荷の配置数')
    parser.add_argument('--pixel_len', type=float, default=1,
                        help='1ピクセルが表す長さ [m]')
    parser.add_argument('--eps0', type=float, default=1,
                        help='真空の誘電率 [Fm-1]')
    args = parser.parse_args()

    mat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    print(approx_surface_potential(mat))

    mesh = create_sphere_surface_mesh(20, 20)

    solve_core(mesh, np.ones(mesh.shape[0]))


if __name__ == "__main__":
    main()
