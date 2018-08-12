import numpy as np
import cv2
from .config import cfg
from ..utils.blob import im_list_to_blob

# 对图片进行了放缩
def _get_image_blob(im):
    # 类型转换
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS # cfg.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    #   [[[-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]
    #     ...
    #     [-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]]
    #     ...
    #     [-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]
    #     [-101.9801 -114.9465 -121.7717]]]


    im_shape = im_orig.shape # (300, 300, 3)
    im_size_min = np.min(im_shape[0:2]) # im_shape[0:2] == (300, 300)
    im_size_max = np.max(im_shape[0:2]) # im_size_min == 300
                                        # im_size_max == 300
    processed_ims = []
    im_scale_factors = []

    # (600,) 以元组的长度为循环次数
    # (600,) 只循环一次
    # traget_size == 600
    for target_size in cfg.TEST.SCALES: # TEST.SCALES = (600,)
        # im_scale为图片放缩倍数
        im_scale = float(target_size) / float(im_size_min) # im_scale == 2.0
        # 防止尺寸超过最大的尺寸限制
        # Prevent the biggest axis from being more than MAX_SIZE
        # round() 方法返回浮点数x的四舍五入值。
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE: # TEST.MAX_SIZE = 1000
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)

        # im变成 600, 600, 3 放大了图片 
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def _get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

# im 是 numpy类型
def test_ctpn(sess, net, im, boxes=None):
    # blob to hold the input images
    # im_scales 是图片放缩倍数
    blobs, im_scales = _get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN: # TEST.HAS_RPN = True
        im_blob = blobs['data'] # blobs['data'] --> 很多个生成的blob
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    # forward pass
    if cfg.TEST.HAS_RPN:
        feed_dict = {net.data: blobs['data'], net.im_info: blobs['im_info'], net.keep_prob: 1.0}

    # VGGnet_test继承了Network类
    rois = sess.run([net.get_output('rois')[0]],feed_dict=feed_dict)
    rois=rois[0]

    scores = rois[:, 0]
    if cfg.TEST.HAS_RPN:
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
    return scores,boxes
