from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机点
np.random.seed(0)
n_points = 100
x = np.random.rand(n_points) * 10
y = np.random.rand(n_points) * 10
z = np.sin(np.sqrt(x**2 + y**2)) + np.random.normal(scale=0.1, size=n_points)

def rbf_interpolation(x, y, z, n_grid=10):
    """
    # 生成一些随机点
    np.random.seed(0)
    n_points = 100
    x = np.random.rand(n_points) * 10
    y = np.random.rand(n_points) * 10
    z = np.sin(np.sqrt(x**2 + y**2)) + np.random.normal(scale=0.1, size=n_points)
    """
    # 构造 3D 网格点
    xi = np.linspace(x.min(), x.max(), n_grid)
    yi = np.linspace(y.min(), y.max(), n_grid)
    xi, yi = np.meshgrid(xi, yi)
    xi_flat = xi.flatten()
    yi_flat = yi.flatten()

    # 计算核矩阵
    X = np.column_stack([x, y])
    K = rbf_kernel(X)

    # 求解线性方程组
    w = np.linalg.solve(K, z)

    # 计算插值值
    X_test = np.column_stack([xi_flat, yi_flat])
    K_test = rbf_kernel(X, X_test)
    zi = np.dot(K_test.T, w).reshape(xi.shape)
    
    return xi, yi, zi

xi, yi, zi = rbf_interpolation(x, y, z, n_grid=10)

def compute_curvature(surface):
    """
    G is the Gaussian curvature and H is the mean curvature,
    G  measures how much the surface is curved in the x and y directions.
    H  measures the average of the two principal curvatures at each point on the surface.
    curvature coefficient K, measures the overall curvature of the surface at each point,
    with positive values indicating convex areas, negative values indicating concave areas, 
    and zero values indicating a flat surface.
    """
    dx, dy = np.gradient(surface)
    dxy, dxx = np.gradient(dx)
    dyy, _ = np.gradient(dy)
    G = (1 + dy ** 2) * dxx - 2 * dx * dy * dxy + (1 + dx ** 2) * dyy
    H = (dx ** 2 + dy ** 2 + 1) ** 1.5
    K = G / H
    
    return K

def is_concave(surface):
    """
    计算曲率半径，并判断曲面的凸凹性
    如果曲率半径中有任意一项小于零，则曲面被认为是凹的，返回True，否则返回False。
    当K> 0时，表示曲面局部上凸出，而K <0则意味着曲面凹陷。K = 0时，表明曲面在该点上是平的。
    K值的绝对值越大，表明该点上面的曲率越大，即弧线的半径越小，曲面越弯曲。因此，K值越大，曲面凸度越大。
    """
    K = compute_curvature(surface)
    R = 1 / K
    print(R.shape)
    sub_n = R.shape[0] // 3
    R_sub = R[sub_n:-sub_n, sub_n:-sub_n]
    print(R_sub.mean(), R_sub.max(), R_sub.min())
    return (R < 0).any()

# k = gaussian_curvature(x, y, z)
k = compute_curvature(zi)
c = is_concave(zi)
print(k.max(), k.min(), c)
def draw_3dcurve(x, y, z, xi, yi, zi, k):
    # 绘制拟合结果和高斯曲率
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(x, y, z, c='k')
    ax.plot_surface(xi, yi, zi, alpha=0.5, cmap='coolwarm')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax2 = fig.add_subplot(122)
    ax2.imshow(k, cmap='coolwarm')
    ax2.set_title('Gaussian Curvature')

    plt.savefig("gausian_curve.png")
    
draw_3dcurve(x, y, z, xi, yi, zi, k)
