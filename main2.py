import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay


def create_sphere_surface_mesh(n_lon, n_lat):
    """
    球面をメッシュによって近似し、三角化を施した結果を返す

    返り値は (メッシュ数, 3角形頂点, xyz) の ndarray
    """
    u = np.linspace(0, 2 * np.pi, n_lon + 1)
    v = np.linspace(-1, 1, n_lat + 1)
    u, v = np.meshgrid(u, v)
    u = u.flatten()
    v = v.flatten()

    t = np.arcsin(v)
    r = np.cos(t)
    x = r * np.cos(u)
    y = r * np.sin(u)
    z = v

    tri = Delaunay(np.vstack([u, v]).T)

    p = np.vstack([x, y, z]).T

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2])
    fig.savefig("a.png")

    # 面積0の縮退したメッシュは除く
    mesh = p[tri.simplices]
    mesh = np.array([mat for mat in mesh
                     if not np.isclose(np.linalg.det(mat), 0, atol=1e-6)])

    return mesh


def area(mat):
    """
    三角形の面積
    """
    a = mat[1] - mat[0]
    b = mat[2] - mat[0]
    return np.sqrt(np.dot(a, a) * np.dot(b, b) - np.dot(a, b) ** 2) / 2


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


class Solver:
    """
    与えられた導体形状と外部電場から電荷分布を求める
    """

    def __init__(self, mesh, pot_fun, eps0=1 / (4 * np.pi)):
        self.mesh = mesh
        self.pot_fun = pot_fun
        self.eps0 = eps0
        self.qs = None

    def _prepare(self):
        """
        準備計算
        """
        self.n = self.mesh.shape[0]
        # 重心
        self.gs = self.mesh.mean(axis=1)
        # 表面での単位電荷量に対するポテンシャル
        self.pot_surface = np.array([self.approx_surface_potential(mat)
                                     for mat in self.mesh])

    def approx_surface_potential(self, mat, n_samples=1000):
        """
        xyz空間中の三角形に一様に電荷が分布しているとき、その重心のポテンシャルをモンテカルロ計算する。

        ここでは三角形内の総電荷量は単位量と仮定する。
        """
        ps = sample_triangle_points(mat, n_samples)
        gs = np.mean(mat, axis=0)

        integ_approx = np.mean(1 / np.linalg.norm(ps - gs, axis=1))
        # 面積の項は積分近似と電荷密度の項で打ち消し合うのでここでは現れないことに注意
        return integ_approx / (4 * np.pi * self.eps0)

    def _solve_core(self, pots):
        """
        ポテンシャル条件を満たす荷量を計算する
        """

        # M[i, j] := 位置 j に単位量の電荷があるときの位置 i のポテンシャル
        def distmat2(v):
            xx, yy = np.meshgrid(v, v)
            return (xx - yy) ** 2

        d = np.sqrt(distmat2(self.gs[:, 0]) +
                    distmat2(self.gs[:, 1]) + distmat2(self.gs[:, 2]))
        for i in range(self.n):
            d[i, i] = 1e-18  # 警告が鬱陶しいので非ゼロにしておく
        M = 1 / (4 * np.pi * self.eps0 * d)
        for i in range(self.n):
            M[i, i] = self.pot_surface[i]

        # TODO: 本当は非対角成分で対角より大きい値があったら潰すべきだが、あまりなさそうなので飛ばす
        self.qs = np.linalg.solve(M, pots)
        print(self.qs.sum())

    def solve(self, pot0=0):
        """
        電荷分布を計算する

        Args:
            pot0: 導体表面でのポテンシャル
        """
        self._prepare()

        pots = np.array([self.pot_fun(g) for g in self.gs])
        self._solve_core(pot0 - pots)

    def calc_pot(self, p):
        """
        与えられた点のポテンシャルを計算する (外部電場の影響を考慮して計算する)
        """
        pot = self.pot_fun(p)

        """
        for i in range(self.n):
            r = np.linalg.norm(p - self.gs[i])
            pot += self.qs[i] * min(1 / (4 * np.pi * self.eps0 * r),
                                    self.pot_surface[i])
        """
        rs = np.linalg.norm(p - self.gs, axis=1)
        pot_for_unit = np.minimum(1 / (4 * np.pi * self.eps0 * rs),
                                  self.pot_surface)
        pot += np.dot(self.qs, pot_for_unit)
        return pot


def get_pot_of_const_ef(e0):
    """
    一様な静電場におけるポテンシャル関数を返す
    """
    def pot(p):
        return np.dot(p, e0)
    return pot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_lon', type=int, default=20,
                        help='メッシュの分解能 (経度方向)')
    parser.add_argument('--n_lat', type=int, default=20,
                        help='メッシュの分解能 (緯度方向)')
    parser.add_argument('--eps0', type=float, default=1,
                        help='真空の誘電率 [Fm-1]')
    args = parser.parse_args()

    mesh = create_sphere_surface_mesh(args.n_lon, args.n_lat)
    pot_fun = get_pot_of_const_ef(np.array([0.3, 0, 0]))
    solver = Solver(mesh, pot_fun)

    mat = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    print(solver.approx_surface_potential(mat))

    # solver._solve_core(np.ones(mesh.shape[0]))
    solver.solve()

    for i in range(-20, 21):
        print(i / 10, solver.calc_pot(np.array([i / 10, 0, 0.86])))


if __name__ == "__main__":
    main()
