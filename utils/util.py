
import scipy


def Hotelling_statistic(dist1, dist2):
    """
    This function is for compare two distribution.

    Inputs:
    dist1 and dist2: all in numpy

    Outputs:

    """
    # # print(dist1)
    # # print(dist1.shape)
    # # print(dist1.shape[1])
    # # print(type(dist1.shape[1]))
    # d = dist1.shape[1]
    # mean1 = np.mean(dist1,axis=0)
    # mean2 = np.mean(dist2,axis=0)
    # # print(mean1.size())
    # cov1 = np.cov(dist1, rowvar=False)
    # cov2 = np.cov(dist2, rowvar=False)
    # pooled_cov = ((len(dist1) - 1) * cov1 + ((len(dist2)) - 1) * cov2) / (len(dist1) + len(dist2) - 2)
    # t2 = scipy.stats.hotelling(mean1, mean2, pooled_cov, len(dist1), len(dist2))
    # df1, df2 = d, len(dist1) + len(dist2) - d - 1
    # p_value = 1 - scipy.stats.f.cdf(df1 / df2 * t2, df1, df2)

    print(scipy.stats.ks_2samp(dist1,dist2))
    return 1