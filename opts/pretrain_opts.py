from collections import OrderedDict

pretrain_opts = OrderedDict()


"""
    imagenet_refine.pkl data structure:
        key-level1: video (relative) directory
        key-level2: 
            images: a list contains all image names
            gt: a list contains all boxes (trackid==0), format is (minx, miny, w, h)
"""
pretrain_opts['vid_pkl'] = "./VID/imagenet_refine.pkl"
pretrain_opts['vid_home'] = "./VID/ILSVRC2015/Data/VID/train"
pretrain_opts['model_path'] = './snapshots/rt_mdnet_fclayers.pth'

pretrain_opts['batch_frames'] = 8
pretrain_opts['batch_pos'] = 32
pretrain_opts['batch_neg'] = 96

pretrain_opts['overlap_pos'] = [0.7, 1]
pretrain_opts['overlap_neg'] = [0, 0.5]

pretrain_opts['opn_imsize'] = (864, 480)      # w, h, can be divided by 32
pretrain_opts['mean_value'] = [102.9801, 115.9465, 122.7717]

pretrain_opts['lr'] = 0.0001
#pretrain_opts['lr'] = 0.001
pretrain_opts['w_decay'] = 0.0005
pretrain_opts['momentum'] = 0.9
pretrain_opts['grad_clip'] = 10
pretrain_opts['ft_layers'] = ['fc']
pretrain_opts['lr_mult'] = {'fc': 1}
#pretrain_opts['n_cycles'] = 1000
pretrain_opts['n_cycles'] = 200
pretrain_opts['seqbatch_size'] = 50
pretrain_opts['verbose_freq'] = 10
pretrain_opts['valid_freq'] = 5000
pretrain_opts['log_file'] = "./dashboard/train_drsn.log"
pretrain_opts['drsn_n_cycles'] = 50
pretrain_opts['dump_freq'] = 10000
pretrain_opts['drsn_snapshot_home'] = "drsn_snapshot"
pretrain_opts['drsn_seqbatch_size'] = 4

##################################### from RCNN #############################################
pretrain_opts['padding'] = 1.2
pretrain_opts['padding_ratio']=5.
#pretrain_opts['padded_img_size'] = pretrain_opts['img_size']*int(pretrain_opts['padding_ratio'])
pretrain_opts['frame_interval'] = 2
