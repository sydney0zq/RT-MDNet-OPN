import numpy as np
from PIL import Image
import os
import cv2

import torch.utils.data as data
from op.mask_op import get_mask_bbox

def resize_box(insize, outsize, minxy_wh):
    in_w, in_h = insize
    out_w, out_h = outsize
    minx, miny, w, h = minxy_wh
    scale_w, scale_h = out_w*1. / in_w, out_h*1. / in_h
    new_minx, new_miny = int(np.round(minx * scale_w)), int(np.round(miny * scale_h))
    new_w, new_h = int(np.round(w * scale_w)), int(np.round(h * scale_h))
    return np.array([new_minx, new_miny, new_w, new_h])

def overlay_box(image, box, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA):
    if len(box.shape) == 1:
        box = np.array([box])
    for i_box in box:
        minx, miny, maxx, maxy = i_box
        if int(cv2.__version__[0]) == 4:
            image = cv2.rectangle(image, (int(minx), int(miny)), (int(maxx), int(maxy)), \
                    color=color, thickness=thickness, lineType=lineType)
        else:
            print ("cv2 version is lower than 4.0, be careful here...")
            cv2.rectangle(image, (int(minx), int(miny)), (int(maxx), int(maxy)), \
                    color=color, thickness=thickness, lineType=lineType)
    return image


class VIDRegionDataset(data.Dataset):
    def __init__(self, img_dir, img_list, gt, opts):
        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
        self.gt = gt

        self.batch_frames = opts['batch_frames']
        self.opn_imsize = opts['opn_imsize']
        self.mean_value = np.array(opts['mean_value'])

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.imsize = image.size

        #self.interval = pretrain_opts['frame_interval']

    def __iter__(self):
        return self

    def __next__(self):
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        img_pool, box_pool = [], []
        for (img, box) in zip(self.img_list[idx], self.gt[idx]):
            cur_img = np.array(Image.open(img).convert('RGB').resize(self.opn_imsize))
            cur_img = cv2.cvtColor(cur_img, cv2.COLOR_RGB2BGR).transpose((2, 0, 1))
            cur_box = resize_box(self.imsize, self.opn_imsize, box)
            img_pool.append(cur_img)
            box_pool.append(cur_box)
        img_pool = np.array(img_pool, dtype='float')
        img_pool = img_pool - self.mean_value.reshape((1, 3, 1, 1))
        box_pool = np.array(box_pool, dtype='int')
        return img_pool, box_pool

    def DEBUG_imgbox(self, img_pool, box_pool, savedir):
        os.makedirs(savedir, exist_ok=True)
        img_pool = img_pool + self.mean_value.reshape((1, 3, 1, 1))
        img_pool = img_pool.transpose((0, 2, 3, 1))
        for i, (img, box) in enumerate(zip(img_pool, box_pool)):
            imgbox = overlay_box(img, np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]]), 
                                 color=[220, 100, 230], thickness=3)
            cv2.imwrite("{}/{:05d}.jpg".format(savedir, i), imgbox)
        print ("DEBUG imgbox done...")


from DAVIS.api import DAVIS_API, cross2otb, otb2cross, overlap_ratio, get_mask_bbox
from PIL import Image
readobjmask = lambda x, obj_id: (np.array(Image.open(x)) == int(obj_id)).astype(np.float)

class DAVISRegionDataset:
    def __init__(self, opts, data_preprocess=True):
        self.davis = DAVIS_API()
        self.seq_list = self.davis.vdnames
        self.data_preprocess = data_preprocess
        self.img_size = opts['img_size']
        self.mean_value = np.array(opts['mean_value'])
        
        table = []
        for seq_name in self.seq_list:
            label_ids = self.davis.get_label_ids(seq_name)
            table += [ "{}/{}".format(seq_name, label_id) for label_id in label_ids ]
        self.table = table

    def __getitem__(self, index):
        seq_name, label_id = self.table[index].split('/')
        return self.query(seq_name, label_id, self.data_preprocess)

    def query(self, seq_name, label_id, preprocess=True):
        img_list = self.davis.get_imglist(seq_name)
        anno_list = self.davis.get_annolist(seq_name)
        gt_list = []
        for anno in anno_list:
            gt = get_mask_bbox(readobjmask(anno, label_id))
            #gt = cross2otb(np.array(gt))
            gt_list.append(gt)
        gt_list = np.array(gt_list)

        if preprocess is True:
            imgs, boxes = self.preprocess(img_list, gt_list)
            return imgs, boxes, seq_name, label_id
        else:
            return img_list, gt_list, seq_name, label_id

    def preprocess(self, img_list, gt_list):
        imgs, boxes = [], []

        imsize = Image.open(img_list[0]).convert('RGB').size
        for (img_path, box) in zip(img_list, gt_list):
            img = np.array(Image.open(img_path).convert('RGB').resize(self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).transpose((2, 0, 1))
            box = resize_box(imsize, self.img_size, box)
            imgs.append(img)
            boxes.append(box)
        imgs = np.array(imgs, dtype='float')
        imgs = imgs - self.mean_value.reshape((1, 3, 1, 1))
        boxes = np.array(boxes, dtype='int')

        return imgs, boxes

    def __len__(self):
        return len(self.table)

from YVOS.api import YVOS_API

class YVOSRegionDataset:
    def __init__(self, opts, split=None, data_preprocess=True):
        self.yvos = YVOS_API("train_all_frames")
        self.seq_list = self.yvos.vdnames
        self.data_preprocess = data_preprocess
        self.img_size = opts['opn_imsize']
        self.mean_value = np.array(opts['mean_value'])
        
        table = []
        for seq_name in self.seq_list:
            label_ids = self.yvos.get_label_ids(seq_name)
            table += [ "{}/{}".format(seq_name, label_id) for label_id in label_ids ]
        
        assert split is not None, "You must specify which split"
        if split == "train":
            self.table = table[:5500]
        elif split == "val":
            self.table = table[5500:]
        elif split == "all":
            self.table = table
        

    def __getitem__(self, index):
        seq_name, label_id = self.table[index].split('/')
        return self.query(seq_name, label_id, self.data_preprocess)

    def query(self, seq_name, label_id, preprocess=True):
        img_list = self.yvos.get_imglist(seq_name, label_id)
        anno_list = self.yvos.get_annolist(seq_name, label_id)

        if preprocess is True:
            imgs, masks = self.preprocess(img_list, anno_list, label_id)
            return imgs, masks, seq_name, label_id
        else:
            return img_list, anno_list, seq_name, label_id

    def preprocess(self, img_list, anno_list, label_id):
        imgs, masks = [], []

        imsize = Image.open(img_list[0]).convert('RGB').size
        for (img_path, anno_path) in zip(img_list, anno_list):
            img = np.array(Image.open(img_path).convert('RGB').resize(self.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).transpose((2, 0, 1))
            mask = np.array(Image.open(anno_path).resize(self.img_size, resample=Image.NEAREST)) == int(label_id)
            mask = mask[None, :]
            imgs.append(img)
            masks.append(mask)
        imgs = np.array(imgs, dtype='float')
        imgs = imgs - self.mean_value.reshape((1, 3, 1, 1))
        masks = np.array(masks, dtype='uint8')

        return imgs, masks

    def __len__(self):
        return len(self.table)

    def DEBUG_imgbox(self, img_pool, mask_pool, savedir):
        os.makedirs(savedir, exist_ok=True)
        img_pool = img_pool + self.mean_value.reshape((1, 3, 1, 1))
        img_pool = img_pool.transpose((0, 2, 3, 1))
        mask_pool = mask.pool.transpose((0, 2, 3, 1))
        for i, (img, mask) in enumerate(zip(img_pool, mask_pool)):
            imgbox = overlay_mask(img, np.array([box[0], box[1], box[0]+box[2], box[1]+box[3]]), 
                                 color=[220, 100, 230], thickness=3)
            cv2.imwrite("{}/{:05d}.jpg".format(savedir, i), imgbox)
        print ("DEBUG imgbox done...")

if __name__ == "__main__":
    from opts.pretrain_opts import pretrain_opts
    import torch
    #davis = DAVISRegionDataset()
    #yvos = YVOSRegionDataset(pretrain_opts, split="train")
    yvos = YVOSRegionDataset(pretrain_opts, split="all")
    
    for i in [178, 1505, 1616, 2743]:
        imgs, masks, seq_name, label_id = yvos[i]
        for j, i_mask in enumerate(masks):
            i_mask = i_mask[0]
            box = get_mask_bbox(i_mask)
            minx, miny, maxx, maxy = [int(x) for x in box]
            if (minx >= maxx or miny >= maxy):
                print (seq_name, label_id, j)
                print (minx, miny, maxx, maxy)

    exit()

    def collate_fn_(data):
        return data[0]

    import time
    ts = time.time()
    for i in range(100):
        yvos[i]
    ts_ = time.time()
    print (ts_ - ts)

    dl = torch.utils.data.DataLoader(yvos, batch_size=1, shuffle=False, num_workers=8, collate_fn=collate_fn_)
    import time
    ts = time.time()
    i = 0
    for item in dl:
        #time.sleep(1)
        i += 1
        if i == 100:
            break
    ts_ = time.time()
    print (ts_ - ts)




