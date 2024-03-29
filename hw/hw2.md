# HW2

PB20111689 蓝俊玮

## 3.1

原因：数字图像是离散的。直方图均衡化方法是一对一或者多对一的映射关系，即原图像的某一灰度级或者某几个灰度级只能映射为均衡化图像的一个灰度级，因此不能实现理想的均衡。

## 3.2

当原始图像概率密度函数为连续情况下。有：
$$
\frac{ds}{dr}=\frac{p_r(r)}{p_s(s)}
$$
变换函数为：$s=\int_{0}^{r}p_r(x)dx$，均衡化处理之后要求图像灰度分布的概率密度函数 $p_s(s)\equiv 1$，即均衡化处理之后要求灰度分布为均匀分布。

如果此时对该增强后的数字图像再次进行增强，则有变换函数为：$s'=\int_{0}^{s}p_s(x)dx$，由于 $p_s(s)\equiv1$，所以可以得出再次增强后的灰度分布概率密度函数为：$p_{s'}(s')\equiv1$，与之前的 $p_s(s)\equiv1$ 是一样的。因此不会改变其结果。

当原始图像概率密度函数为离散情况下。有变换函数：
$$
s_k=\sum_{i=0}^{k}\frac{n_i}{n}
$$
然后对其进行取整：
$$
S_k=\rm{int}[(L-1)s_k+0.5]
$$

则得到的新的灰度级在 $[0,L-1]$ 区间之内。则再一次进行均衡化时，则有：
$$
s_{S_k}=\sum_{i=0}^{S_k}\frac{n_i'}{n}
$$
其中 $s_{S_k}$ 表示的由第一次均衡化之后得到的灰度级 $S_k$  再次进行均衡化之后得到的灰度级。而 $n_i'$ 表示的则是第一次均衡化之后灰度级为 $i$ 的像素个数。由于均衡化的性质，我们知道变换前后其累计直方图的值是不变的，因此可以得到：
$$
s_{S_k}=\sum_{i=0}^{S_k}\frac{n_i'}{n}=\sum_{i=0}^{k}\frac{n_i}{n}=s_k
$$
则对其进行取整后发现：
$$
S_{S_k}=\rm{int}[(L-1)s_{S_k}+0.5]=\rm{int}[(L-1)s_k+0.5]=S_k
$$
所以再次进行均衡化之后，灰度级是不会变化的。因此是不会改变其结果的。

## 3.3

对其列表：

| 序号 |         运算         |         |         |         |         |         |         |         |         |
| :--: | :------------------: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: | :-----: |
|  1   | 列出原始灰度级 $s_k$ |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
|  2   |    计算原始直方图    |  0.174  |  0.088  |  0.086  |  0.08   |  0.068  |  0.058  |  0.062  |  0.384  |
|  3   |  计算原始累积直方图  |  0.174  |  0.262  |  0.348  |  0.428  |  0.496  |  0.554  |  0.616  |  1.000  |
|  4   |      规定直方图      |    0    |   0.4   |    0    |    0    |   0.2   |    0    |    0    |   0.4   |
|  5   |  计算规定累计直方图  |    0    |   0.4   |   0.4   |   0.4   |   0.6   |   0.6   |   0.6   |   1.0   |
|  6   |         SML          |    1    |    1    |    1    |    1    |    1    |    4    |    4    |    7    |
|  7   |       映射关系       | $0\to1$ | $1\to1$ | $2\to1$ | $3\to1$ | $4\to1$ | $5\to4$ | $6\to4$ | $7\to7$ |
|  8   |     变换后直方图     |    0    |  0.496  |    0    |    0    |  0.120  |    0    |    0    |  0.384  |

## 3.4

$n\times n$ 中值滤波器的程序如下：

```matlab
img = imread("lena.bmp");
img_salt_pepper = imnoise(img, "salt & pepper");
img_gauss = imnoise(img, "gaussian");
n = 3;
[x, y] = size(img);
img_result1 = img;
img_result2 = img;
for i = 1 : x - (n - 1)  
    for j = 1 : y - (n - 1)  
        img_result1(i + (n - 1) / 2, j + (n - 1) / 2) = median(img_salt_pepper(i : i + (n - 1), j : j + (n - 1)), 'all');
        img_result2(i + (n - 1) / 2, j + (n - 1) / 2) = median(img_gauss(i : i + (n - 1), j : j + (n - 1)), 'all');
    end  
end
subplot(2, 2, 1); imshow(img_salt_pepper);
subplot(2, 2, 2); imshow(img_result1);
subplot(2, 2, 3); imshow(img_gauss);
subplot(2, 2, 4); imshow(img_result2);
```

上述程序的运行结果如下，可以看出中值滤波器对椒盐噪声的处理效果比较好，而对高斯噪声的处理效果比较差。

![](D:\hwset\digital-image\median_filter.png)

当模板中心移过图像中每个位置时，可以采用下述方法快速地更新中值：

因为 $n\times n$ 的模板在横向移动时，每次移动只有 $n$ 个像素的值会发生变化，因此可以使用一个队列存储该 $n\times n$ 个像素。在模板移动的时候，前面的 $n$ 个像素值出队。而将新加入的 $n$ 个像素值加入到队尾。由于在之前的操作过程中，我们已经得到了出队之前的 $n\times n$ 个像素的有序序列，因此在前 $n$ 个像素出队之后，我们依然还有 $n\times n-n$ 个像素的有序序列。因此我们只需要对新加入的 $n$ 个像素值调用插入排序，这样在前面的像素值已经有序的情况下，采用插入排序的时间复杂度可以达到最优。所以在调用完插入排序之后，便可以快速地获得中值。
