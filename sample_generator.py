import numpy as np
import random


# bbox should be in (4,) with (minx, miny, maxx, maxy) format
def gen_samples_4tinybox(bbox, n, scale_range=(2, 10)):
    minx, miny, maxx, maxy = bbox[0], bbox[1], bbox[2], bbox[3]
    w, h = max(maxx-minx, 1), max(maxy-miny, 1)
    cx, cy = (maxx+minx)/2., (maxy+miny)/2.
    rand_box = np.zeros((n, 4))
    for i in range(n):
        new_w = w*random.randint(scale_range[0], scale_range[1])
        new_h = h*random.randint(scale_range[0], scale_range[1])
        new_minx = cx-new_w/2.
        new_miny = cy-new_h/2.
        new_maxx = cx+new_w/2.
        new_maxy = cy+new_h/2.
        rand_box[i] = np.array([new_minx, new_miny, new_maxx, new_maxy])
    return rand_box


# Input box should in (minx, miny, maxx, maxy)
def overlap_ratio_crossbox(rect1, rect2):
    if rect1.ndim == 1: rect1 = rect1[None, :]
    if rect2.ndim == 1: rect2 = rect2[None, :]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,2], rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,3], rect2[:,3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = (rect1[:,2]-rect1[:,0])*(rect1[:,3]-rect1[:,1]) + (rect2[:,2]-rect2[:,0])*(rect2[:,3]-rect2[:,1]) - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou

def gen_samples(generator, bbox, n, overlap_range=None, scale_range=None):

    if overlap_range is None and scale_range is None:
        return generator(bbox, n)

    else:
        samples = None
        remain = n
        factor = 2
        while remain > 0 and factor < 64:
            remain = int(remain)
            samples_ = generator(bbox, remain*factor)

            idx = np.ones(len(samples_), dtype=bool)
            if overlap_range is not None:
                r = overlap_ratio_crossbox(samples_, bbox)
                idx *= (r >= overlap_range[0]) * (r <= overlap_range[1])
            if scale_range is not None:
                s = np.prod(samples_[:,2:], axis=1) / np.prod(bbox[2:])
                idx *= (s >= scale_range[0]) * (s <= scale_range[1])

            samples_ = samples_[idx,:]
            samples_ = samples_[:min(remain, len(samples_))]
            if samples is None:
                samples = samples_
            else:
                samples = np.concatenate([samples, samples_])
            remain = n - len(samples)
            factor = factor*2

        return samples


class SampleGenerator():
    def __init__(self, type, img_size, trans_f=1, scale_f=1, aspect_f=None, valid=False):
        self.type = type
        self.img_size = np.array(img_size) # (w, h)
        self.trans_f = trans_f
        self.scale_f = scale_f
        self.aspect_f = aspect_f
        self.valid = valid

    # input bb is Nx4 with (minx, miny, maxx, maxy) format
    def __call__(self, bb, n):
        bb = np.array(bb, dtype='float32')
        n = int(n)

        # (minx, miny, maxx, maxy) to (center_x, center_y, w, h)
        sample = np.array([(bb[0]+bb[2])/2, (bb[1]+bb[3])/2, bb[2]-bb[0], bb[3]-bb[1]], dtype='float32')
        samples = np.tile(sample[None,:], (n,1))

        # vary aspect ratio
        if self.aspect_f is not None:
            ratio = np.random.rand(n, 1)*2 - 1
            samples[:, 2:] *= self.aspect_f ** np.concatenate([ratio, -ratio], axis=1)

        # sample generation
        if self.type=='gaussian':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * np.clip(0.5*np.random.randn(n,2),-1,1)
            samples[:,2:] *= self.scale_f ** np.clip(0.5*np.random.randn(n,1),-1,1)

        elif self.type=='uniform':
            samples[:,:2] += self.trans_f * np.mean(bb[2:]) * (np.random.rand(n,2)*2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        elif self.type=='whole':
            m = int(2*np.sqrt(n))
            xy = np.dstack(np.meshgrid(np.linspace(0,1,m),np.linspace(0,1,m))).reshape(-1,2)
            xy = np.random.permutation(xy)[:n]
            samples[:,:2] = bb[2:]/2 + xy * (self.img_size-bb[2:]/2-1)
            samples[:,2:] *= self.scale_f ** (np.random.rand(n,1)*2-1)

        # adjust bbox range
        samples[:,2:] = np.clip(samples[:,2:], 5, self.img_size-5.)
        if self.valid:
            samples[:,:2] = np.clip(samples[:,:2], samples[:,2:]/2, self.img_size-samples[:,2:]/2-1)
        else:
            samples[:,:2] = np.clip(samples[:,:2], 0, self.img_size)

        # (center_x, center_y, w, h) to (minx, miny, w, h) to (minx, miny, maxx, maxy)
        samples[:,:2] -= samples[:,2:]/2
        samples[:,2:] += samples[:,:2]

        return samples

    def set_trans_f(self, trans_f):
        self.trans_f = trans_f

    def get_trans_f(self):
        return self.trans_f


if __name__ == "__main__":
    bbox = np.array([0, 0, 7, 4])
    b = gen_samples_4tinybox(bbox, 100)
    print (b)


    exit()
    img_size = (864, 480)
    sg =  SampleGenerator(type="gaussian", img_size=img_size, trans_f=0.1, scale_f=1.1)
    #box = sg(np.array([10, 10, 50, 50]), 10)
    i = 0
    while True:
        box = gen_samples(sg, np.array([10, 10, 20, 20]), 10, [0.7, 1])
        i += 1
        if i % 10000 == 0: print (i)
        if len(box) != 10:
            print (i)
            break
    print (box)
    print (len(box))

    





    

