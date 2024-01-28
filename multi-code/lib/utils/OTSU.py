import numpy as np
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False


def calc_grayhist(image, max_length=255):
    # 无论多少维数组都展成一维
    image_flatten = image.flatten()
    image_flatten = np.array(image_flatten, dtype=np.int32)
    # plt.hist(image_flatten, bins=max_length+1, color="g", histtype="bar", rwidth=1, alpha=0.6)
    # plt.show()
    # print(image_flatten.shape[0])
    # 求出像素值0~max_length的统计直方图
    grayhist = np.bincount(image_flatten)
    # 归一化直方图
    grayhist = grayhist / image_flatten.shape[0]
    return grayhist


def get_multi_threshold(grey_img, m=2, max_length=255):
    """
    返回阈值 [0,T0) [T0,T1) [T1,255]
    Args:
        grey_img: 灰度图
        m: 阈值数

    Returns:
    """
    # 先获得像素的分布直方图
    grayhist = calc_grayhist(grey_img, max_length=max_length)
    grayhist[0] = 0.0
    # 计算全局平均灰度值
    mG = np.array([i*p for i, p in enumerate(grayhist)]).sum()
    # print(mG)
    # 预先计算存储最后一段的概率后缀和与灰度值期望后缀和
    w2_array = np.zeros([max_length+2])
    w2_array[:-1] = grayhist.copy()
    m2_array = w2_array.copy()
    for i in range(max_length, -1, -1):
        w2_array[i] += w2_array[i+1]
        m2_array[i] = m2_array[i+1] + m2_array[i] * i

    if m == 2:
        T0 = T1 = 0
        max_sigmaB2 = 0
        w0 = 0.0
        m0 = 0.0
        for temp0 in range(1, max_length):
            # 第一段的概率和与灰度值期望可以累加，减少不必要的计算量
            w0 += grayhist[temp0-1]
            m0 += grayhist[temp0-1] * (temp0-1)
            # 计算u0和第一段方差
            u0 = m0 / (w0 + 1e-18)
            sigma0 = w0 * (u0 - mG) ** 2

            w1 = 0.0
            m1 = 0.0
            for temp1 in range(temp0+1, max_length+1):
                # 第二段的概率和与灰度值期望也可以累加，减少不必要的计算量
                w1 += grayhist[temp1-1]
                m1 += grayhist[temp1-1] * (temp1-1)
                # 获得最后一段的w2、m2
                w2 = w2_array[temp1]
                m2 = m2_array[temp1]
                # 计算u1和u2,第二段和第三段方差
                u1 = m1 / (w1 + 1e-18)
                u2 = m2 / (w2 + 1e-18)
                sigma1 = w1 * (u1 - mG) ** 2
                sigma2 = w2 * (u2 - mG) ** 2
                sigmaB2 = sigma0 + sigma1 + sigma2
                # if temp0 < 50:
                #     print("T0:{},T1:{},sigmaB2:{}".format(temp0, temp1, sigmaB2))
                if sigmaB2 >= max_sigmaB2:
                    max_sigmaB2 = sigmaB2
                    T0 = temp0
                    T1 = temp1
        # print(max_sigmaB2)
        return T0, T1
    elif m == 1:
        pass
    else:
        raise Exception("don't support number of threshold!")





