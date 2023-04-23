# HW1

PB20111689 蓝俊玮

## 2.1

由空间点坐标 $(X,Y,Z)=(1,2,3)$ 和 $\lambda=0.5$ 的镜头，根据成像几何模型可以得到：

摄像机坐标为：$(\frac{\lambda X}{\lambda-Z},\frac{\lambda Y}{\lambda-Z},\frac{\lambda Z}{\lambda-Z})=(-0.2,-0.4,-0.6)$

图像平面坐标为：$(\frac{\lambda X}{\lambda-Z},\frac{\lambda Y}{\lambda-Z})=(-0.2,-0.4)$

## 2.2

- 由 4-连接定义，需要两个像素在“上下左右”处相连。可以发现子集 $S$ 和子集 $T$ 不符合这个条件，因此子集 $S$ 和子集 $T$ 不是 4-连通的。

- 由 8-连接定义，需要两个像素在周围相连。可以看出子集 $S$ 和子集 $T$ 所有像素点值为 1 的周围都有一个像素点的值为 1，因此子集 $S$ 和子集 $T$ 是 8-连通的。

- 由 m-连接定义，需要满足其中一个条件即可：

  - “r 在 $N_4(p)$ 中”
  - “r 在 $N_D(p)$ 中且 $N_4(p)\cap N_4(r)=\empty$”

  通过判断，可以得知子集 $S$ 和子集 $T$ 所有像素点值为 1 的点周围都有符合 m-连接的像素点，因此子集 $S$ 和子集 $T$ 是 m-连通的。

## 2.3

- 当 $v=\{0,1\}$ 时：
  - 无 $D_4$ 通路
  - $D_8$ 通路的最短长度是 4
  - $D_m$ 通路的最短长度是 5
- 当 $v=\{1,2\}$ 时：
  - $D_4$ 通路的最短长度是 6
  - $D_8$ 通路的最短长度是 4
  - $D_m$ 通路的最短长度是 6

## 2.4

图像顺时针旋转 $\theta=45\degree$ 的变换矩阵：
$$
\begin{bmatrix}
\cos\theta& \sin\theta& 0\\
-\sin\theta& \cos\theta& 0\\
0& 0& 1
\end{bmatrix}=
\begin{bmatrix}
\frac{\sqrt{2}}{2}& \frac{\sqrt{2}}{2}& 0\\
-\frac{\sqrt{2}}{2}& \frac{\sqrt{2}}{2}& 0\\
0& 0& 1
\end{bmatrix}
$$
利用该矩阵旋转图像点 $(x,y)=(1,0)$，则有：
$$
\begin{bmatrix}
\frac{\sqrt{2}}{2}& \frac{\sqrt{2}}{2}& 0\\
-\frac{\sqrt{2}}{2}& \frac{\sqrt{2}}{2}& 0\\
0& 0& 1
\end{bmatrix}
\begin{bmatrix}
1\\0\\1
\end{bmatrix}=
\begin{bmatrix}
\frac{\sqrt{2}}{2}\\-\frac{\sqrt{2}}{2}\\1
\end{bmatrix}
$$
所以可以得到旋转后的坐标为：$(x',y')=(\frac{\sqrt{2}}{2},-\frac{\sqrt{2}}{2})$

## 2.5

- 先平移变换后尺度变换：
  $$
  \begin{bmatrix}
  4& 0& 0& 0\\
  0& 3& 0& 0\\
  0& 0& 2& 0\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  1& 0& 0& 2\\
  0& 1& 0& 4\\
  0& 0& 1& 6\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  1\\
  2\\
  3\\
  1
  \end{bmatrix}=\begin{bmatrix}
  4& 0& 0& 0\\
  0& 3& 0& 0\\
  0& 0& 2& 0\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  3\\
  6\\
  9\\
  1
  \end{bmatrix}=
  \begin{bmatrix}
  12\\
  18\\
  18\\
  1
  \end{bmatrix}
  $$
  最后得到的结果是 $(x',y',z')=(12,18,18)$

- 先尺度变换后平移变换：
  $$
  \begin{bmatrix}
  1& 0& 0& 2\\
  0& 1& 0& 4\\
  0& 0& 1& 6\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  4& 0& 0& 0\\
  0& 3& 0& 0\\
  0& 0& 2& 0\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  1\\
  2\\
  3\\
  1
  \end{bmatrix}=
  \begin{bmatrix}
  1& 0& 0& 2\\
  0& 1& 0& 4\\
  0& 0& 1& 6\\
  0& 0& 0& 1
  \end{bmatrix}
  \begin{bmatrix}
  4\\
  6\\
  6\\
  1
  \end{bmatrix}=
  \begin{bmatrix}
  6\\
  10\\
  12\\
  1
  \end{bmatrix}
  $$
  最后得到的结果是 $(x',y',z')=(6,10,12)$

两种计算方式得到的结果是不相同的，我认为是因为每次变换时的“力度”不同。先进行平移变换后，会让尺度变换的效果变得更明显，而先进行尺度变换，则在原来的基础上，尺度变换的“力度”不够大，所以结果差距会比较大。

## 2.6

因为只有两个三角形的顶点作为对应的控制点，即只有 3 对控制点 $(x_1,y_1),(x_2,y_2),(x_3,y_3)$ 和 $(x_1',y_1'),(x_2',y_2'),(x_3',y_3')$。因此符合线性失真的要求。线性失真的空间变换式为：

原图像：$f(x,y)$，失真图像：$g(x',y')$，$x'=a_1x+a_2y+a_3,\ y'=b_1x+b_2y+b_3$

则有：
$$
\begin{bmatrix}
x_1'\\
x_2'\\
x_3'
\end{bmatrix}=
\begin{bmatrix}
x_1& y_1& 1\\
x_2& y_2& 1\\
x_3& y_3& 1
\end{bmatrix}
\begin{bmatrix}
a_1\\
a_2\\
a_3
\end{bmatrix},\quad\begin{bmatrix}
y_1'\\
y_2'\\
y_3'
\end{bmatrix}=
\begin{bmatrix}
x_1& y_1& 1\\
x_2& y_2& 1\\
x_3& y_3& 1
\end{bmatrix}
\begin{bmatrix}
b_1\\
b_2\\
b_3
\end{bmatrix}
$$
这就是线性失真情况下相对应的校正几何形变的空间变换式。