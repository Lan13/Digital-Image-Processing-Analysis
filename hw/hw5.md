# HW5

PB20111689 蓝俊玮

## 6.1

假设物体在 x 和 y 方向上任意的匀速运动方程为：$x_0(t)=at/T$，$y_0(t)=bt/T$，则转移函数 $H(u,v)$ 计算如下：

$$
\begin{align}
H(u,v)=&\ \int_{0}^{T}\exp\bigg(-j2\pi\bigg[ux_0(t)+vy_0(t)\bigg]\bigg)dt\\
=&\ \int_{0}^{T}\exp\bigg(-j2\pi\bigg[\frac{au+bv}{T}t\bigg]\bigg)dt\\
=&\ \frac{T}{j2\pi(au+hv)}\bigg(1-e^{-j2\pi(au+bv)}\bigg)
\end{align}
$$

所以 $H(u,v)$ 就是运动模糊的转移函数。

## 6.2

由已知条件，我们知道物体在 x 方向上的匀加速运动方程为 $x_0(t)=at^2/2$，那么就有 $y_0(t)=0$，则转移函数 $H(u,v)$ 计算如下：
$$
\begin{align}
H(u,v)=&\ \int_{0}^{T}\exp\bigg(-j2\pi\bigg[ux_0(t)+vy_0(t)\bigg]\bigg)dt\\
=&\ \int_{0}^{T}\exp\bigg(-j\pi aut^2\bigg)dt\\
=&\ \int_{0}^{T}\bigg[\cos(\pi aut^2)-j\sin(\pi aut^2)\bigg]dt
\end{align}
$$
所以 $H(u,v)$ 就是运动模糊的转移函数。

匀速运动和匀加速运动所造成的模糊的不同特点：匀速运动所造成的模糊特点是由于物体在相邻帧之间发生了微小的位移，所造成的模糊通常是线性的，模糊程度在运动各处是相同的。匀加速运动所造成的模糊特点是由于物体在运动过程中速度的变化，所造成的模糊通常是非线性的，模糊程度会随着运动速度的增加而增加。

## 6.3

图像模糊的转移函数 $H(u,v)$ 的表示如下：
$$
H(u,v)=\exp\bigg(-\frac{u^2+v^2}{2\sigma^2}\bigg)
$$
若噪声可以忽略，则 $p_n(u,v)=0$，那么恢复这类模糊的维纳滤波方程为：
$$
\begin{align}
\hat{F}(u,v)=&\ \bigg[\frac{H^*(u,v)}{|H(u,v)|^2+\gamma[p_n(u,v)/p_f(u,v)]}\bigg]G(u,v)\\
=&\ \bigg[\frac{\exp(-\frac{u^2+v^2}{2\sigma^2})}{\exp^2(-\frac{u^2+v^2}{2\sigma^2})}\bigg]G(u,v)\\
=&\ \exp(\frac{u^2+v^2}{2\sigma^2})G(u,v)
\end{align}
$$
所以 $\hat{F}(u,v)$ 就是恢复这类模糊的维纳滤波方程。
