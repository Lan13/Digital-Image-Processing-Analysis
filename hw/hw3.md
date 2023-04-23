# HW3

PB20111689 蓝俊玮

## 4.1

首先从题目要证明的形式得知，这是对应于傅里叶变换的离散形式，因此有：
$$
F(u,v)=\frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)e^{\bigg[-j2\pi(\frac{ux}{N}+\frac{vy}{N})\bigg]}\\
f(x,y)=\frac{1}{N}\sum_{u=0}^{N-1}\sum_{v=0}^{N-1}F(u,v)e^{\bigg[j2\pi(\frac{ux}{N}+\frac{vy}{N})\bigg]}\\
$$
因此可以计算：
$$
\begin{align}
F(u-u_0,v-v_0)=&\ \frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)e^{\bigg[-j2\pi(\frac{(u-u_0)x}{N}+\frac{(v-v_0)y}{N})\bigg]}\\
=&\ \frac{1}{N}\sum_{x=0}^{N-1}\sum_{y=0}^{N-1}f(x,y)e^{\bigg[-j2\pi(\frac{ux}{N}+\frac{vy}{N})\bigg]}e^{\bigg[j2\pi(\frac{u_0x}{N}+\frac{v_0y}{N})\bigg]}\\
=&\ F(u,v)e^{\bigg[j2\pi(\frac{u_0x}{N}+\frac{v_0y}{N})\bigg]}\\
\Leftrightarrow&\ f(x,y)e^{\bigg[j2\pi(\frac{u_0x}{N}+\frac{v_0y}{N})\bigg]}
\end{align}
$$
同理：
$$
\begin{align}
f(x-x_0,y-y_0)=&\ \frac{1}{N}\sum_{u=0}^{N-1}\sum_{v=0}^{N-1}F(u,v)e^{\bigg[j2\pi(\frac{u(x-x_0)}{N}+\frac{v(y-y_0)}{N})\bigg]}\\
=&\ \frac{1}{N}\sum_{u=0}^{N-1}\sum_{v=0}^{N-1}F(u,v)e^{\bigg[j2\pi(\frac{ux}{N}+\frac{vy}{N})\bigg]}e^{\bigg[-j2\pi(\frac{ux_0}{N}+\frac{vy_0}{N})\bigg]}\\
=&\ f(x,y)e^{\bigg[-j2\pi(\frac{ux_0}{N}+\frac{vy_0}{N})\bigg]}\\
\Leftrightarrow&\ F(u,v)e^{\bigg[-j2\pi(\frac{ux_0}{N}+\frac{vy_0}{N})\bigg]}
\end{align}
$$
证明完毕

## 4.2

当 $f(x)$ 为连续可积函数时：
$$
\begin{align}
f(x)\circ f(x)=\int_{-\infty}^{\infty}f^*(\tau)f(x+\tau)d\tau\\
\end{align}
$$
则其自相关函数的傅里叶变换如下：
$$
\begin{align}
F=&\ \int_{-\infty}^{\infty}f(x)\circ f(x)\exp{(-j2\pi ux)}dx\\
=&\ \int_{-\infty}^{\infty}\bigg(\int_{-\infty}^{\infty}f^*(\tau)f(x+\tau)d\tau\bigg)\exp{(-j2\pi ux)}dx\\
=&\ \int_{-\infty}^{\infty}f^*(\tau)\bigg(\int_{-\infty}^{\infty}f(x+\tau)\exp{\big(-j2\pi u(x+\tau)\big)}dx\bigg)\exp{(j2\pi u\tau)}d\tau\\
=&\ \int_{-\infty}^{\infty}f^*(\tau)F(u)\exp{(j2\pi u\tau)}d\tau\\
=&\ F(u)\int_{-\infty}^{\infty}f^*(\tau)\exp{(j2\pi u\tau)}d\tau\\
=&\ F(u)\bigg(\int_{-\infty}^{\infty}f(\tau)\exp{(-j2\pi u\tau)}d\tau\bigg)^*\\
=&\ F(u)F^*(u)=|F(u)|^2
\end{align}
$$
当 $f(x)$ 为离散函数时：
$$
f(x)\circ f(x)=\frac{1}{N}\sum_{n=0}^{N-1}f^*(n)f(x+n)
$$
则其自相关函数的傅里叶变换如下：
$$
\begin{align}
F=&\ \frac{1}{N}\sum_{x=0}^{N-1}f(x)\circ f(x)\exp(-\frac{j2\pi ux}{N})\\
=&\ \frac{1}{N}\sum_{x=0}^{N-1}\bigg(\frac{1}{N}\sum_{n=0}^{N-1}f^*(n)f(x+n)\bigg)\exp(-\frac{j2\pi ux}{N})\\
=&\ \frac{1}{N^2}\sum_{n=0}^{N-1}f^*(n)\bigg(\sum_{x=0}^{N-1}f(x+n)\exp(-\frac{j2\pi u(x+n)}{N})\bigg)\exp(\frac{j2\pi un}{N})\\
=&\ \frac{1}{N}\sum_{n=0}^{N-1}f^*(n)F(u)\exp(\frac{j2\pi un}{N})\\
=&\ F(u)\bigg(\frac{1}{N}\sum_{n=0}^{N-1}f(n)\exp(-\frac{j2\pi un}{N})\bigg)^*\\
=&\ F(u)F^*(u)=|F(u)|^2
\end{align}
$$
因此 $f(x)$ 的自相关函数的傅里叶变换就是 $f(x)$ 的功率谱 $|F(u)|^2$ 

## 4.3

要证明离散傅里叶变换和反变换都是周期函数，则证明：$F(u)=F(u+N)$ 和 $f(x)=f(x+N)$
$$
\begin{align}
F(u+N)=&\ \frac{1}{N}\sum_{x=0}^{N-1}f(x)\exp(-\frac{j2\pi (u+N)x}{N})\\
=&\ \frac{1}{N}\sum_{x=0}^{N-1}f(x)\exp(-\frac{j2\pi ux}{N})\exp(-\frac{j2\pi Nx}{N})\\
=&\ \frac{1}{N}\sum_{x=0}^{N-1}f(x)\exp(-\frac{j2\pi ux}{N})\exp(-j2\pi x)\\
=&\ \frac{1}{N}\sum_{x=0}^{N-1}f(x)\exp(-\frac{j2\pi ux}{N})\\
=&\ F(u)
\end{align}
$$
同理：
$$
\begin{align}
f(x+N)=&\ \sum_{u=0}^{N-1}F(u)\exp(\frac{j2\pi u(x+N)}{N})\\
=&\ \sum_{u=0}^{N-1}F(u)\exp(\frac{j2\pi ux}{N})\exp(\frac{j2\pi uN}{N})\\
=&\ \sum_{u=0}^{N-1}F(u)\exp(\frac{j2\pi ux}{N})\exp(j2\pi u)\\
=&\ \sum_{u=0}^{N-1}F(u)\exp(\frac{j2\pi ux}{N})\\
=&\ f(x)
\end{align}
$$
证明完毕。