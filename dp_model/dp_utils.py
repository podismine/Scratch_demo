import numpy as np
from scipy.stats import norm
def label_smooth(x, bin_range = [14, 94], bin_step = 1, sigma = 1):
    eps = 1e-6
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    x = max(min(x,bin_end),bin_start)
    bin_length = bin_end - bin_start
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + bin_step * np.arange(bin_number)
    v = np.zeros((bin_number,)) + eps
    index = int(x) - bin_start
    v[index] = 0.8
    ratio = (x - 0.8 * bin_centers[index]) / (bin_centers[index] * 2.)
    v[index - 1] = ratio - eps
    v[index + 1] = ratio - eps
    return v, bin_centers

def generate_biolabel(label,sigma = 2, bin_step = 1):
    labelset = np.array([i * bin_step + 40 for i in range(int(44 / bin_step))])
    #print(labelset.shape)
    dis = np.exp(-1/2. * np.power((labelset - label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset

def generate_label(label,sigma = 2, bin_step = 1):
    labelset = np.array([i * bin_step + 14 for i in range(int(84 / bin_step))])
    #print(labelset.shape)
    dis = np.exp(-1/2. * np.power((labelset - label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset

def dicreatize(label):
    bin_range = [14,94]
    bin_step = 2
    sigma = 1

    label = label.cpu().numpy()
    ret_y = np.zeros([label.shape[0], 40])
    ret_c = np.zeros([label.shape[0], 40])
    #print(ret_y.shape, ret_c.shape, "retrun shape")
    for batch in range(label.shape[0]):
        y ,c = num2vect(label[batch,...], bin_range, bin_step, sigma)
        #print(y.shape, c.shape)
        ret_y[batch,:], ret_c[batch, :] = y[0,...], c
    return ret_y, ret_c
    #y, bc = dpu.num2vect(label, bin_range, bin_step, sigma)
def num2vect(x, bin_range = [14, 94], bin_step = 2, sigma = 1):
    """
    v,bin_centers = number2vector(x,bin_range,bin_step,sigma)
    bin_range: (start, end), size-2 tuple
    bin_step: should be a divisor of |end-start|
    sigma:
    = 0 for 'hard label', v is index
    > 0 for 'soft label', v is vector
    < 0 for error messages.
    """
    bin_start = bin_range[0]
    bin_end = bin_range[1]
    bin_length = bin_end - bin_start
    if not bin_length % bin_step == 0:
        print("bin's range should be divisible by bin_step!")
        return -1
    bin_number = int(bin_length / bin_step)
    bin_centers = bin_start + float(bin_step) / 2 + bin_step * np.arange(bin_number)
    if sigma == 0:
        x = np.array(x)
        i = np.floor((x - bin_start) / bin_step)
        i = i.astype(int)
        return i, bin_centers
    elif sigma > 0:
        if np.isscalar(x):
            v = np.zeros((bin_number,))
            for i in range(bin_number):
                x1 = bin_centers[i] - float(bin_step) / 2
                x2 = bin_centers[i] + float(bin_step) / 2
                cdfs = norm.cdf([x1, x2], loc=x, scale=sigma)
                v[i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        else:
            v = np.zeros((len(x), bin_number))
            for j in range(len(x)):
                for i in range(bin_number):
                    x1 = bin_centers[i] - float(bin_step) / 2
                    x2 = bin_centers[i] + float(bin_step) / 2
                    cdfs = norm.cdf([x1, x2], loc=x[j], scale=sigma)
                    v[j, i] = cdfs[1] - cdfs[0]
            return v, bin_centers
        
        
def crop_center(data, out_sp):
    """
    Returns the center part of volume data.
    crop: in_sp > out_sp
    Example: 
    data.shape = np.random.rand(182, 218, 182)
    out_sp = (160, 192, 160)
    data_out = crop_center(data, out_sp)
    """
    in_sp = data.shape
    nd = np.ndim(data)
    x_crop = int((in_sp[-1] - out_sp[-1]) / 2)
    y_crop = int((in_sp[-2] - out_sp[-2]) / 2)
    z_crop = int((in_sp[-3] - out_sp[-3]) / 2)
    if nd == 3:
        data_crop = data[x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    elif nd == 4:
        data_crop = data[:, x_crop:-x_crop, y_crop:-y_crop, z_crop:-z_crop]
    else:
        raise ('Wrong dimension! dim=%d.' % nd)
    return data_crop