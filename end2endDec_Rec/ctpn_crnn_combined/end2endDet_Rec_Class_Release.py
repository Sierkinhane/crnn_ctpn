# ctpn packages
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib.networks.factory import get_network
from lib.fast_rcnn.test import test_ctpn
from lib.utils.timer import Timer
from lib.text_connector.detectors import TextDetector

# crnn packages
import torch
from torch.autograd import Variable
from train_crnn import utils
from train_crnn import dataset
from PIL import Image
import train_crnn.models.crnn as crnn
from train_crnn import alphabets

class det_and_rec(object):
    """docstring for det_And_Rec"""
    def __init__(self):
        # ctpn model
        self.ctpn_model_path = 'trained_models/trained_ctpn_models'
        # path to needed detect image
        self.detect_data_path = os.path.join('detect_data/detect_images')
        # crnn model
        self.crnn_model_path = 'trained_models/trained_crnn_models/latestModel.pth'
        # recognition results
        self.txt_results_path = 'detect_data/txt_results/text_info_results.txt'
        # a character dictionary
        self.alphabet = alphabets.alphabet
        # total class
        self.nclass = len(self.alphabet)+1
        # text proposals
        self.boxes = []
        # tensorflow configures
        self.config=tf.ConfigProto(log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.75
        # tensorflow session
        self.sess = tf.Session(config=self.config)
        # load network
        self.net = get_network("VGGnet_test")
        # load model
        self.saver = tf.train.Saver()
        # crnn network
        self.model = crnn.CRNN(32, 1, self.nclass, 256)
        #

    # 图片resize
    def resize_im(self, im, scale, max_scale=None):
        f=float(scale)/min(im.shape[0], im.shape[1])
        if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
            f=float(max_scale)/max(im.shape[0], im.shape[1])

        return cv2.resize(im, 
        	None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

    # 对获得的坐标由上到下排序
    def sort_list(self, min_y_sort_list):
        cnt=0
        for i in range(len(min_y_sort_list)):
            for j in range(1, len(min_y_sort_list)-i):
                if min_y_sort_list[i][1] > min_y_sort_list[i+j][1]:
                    temp = min_y_sort_list[i]
                    min_y_sort_list[i] = min_y_sort_list[i+j]
                    min_y_sort_list[i+j] = temp
                j+=1
            i+=1

        return min_y_sort_list

    # 根据坐标信息剪裁出文本区域
    def crop_images(self, coordinates, base_name):
        img = Image.open(self.detect_data_path+'/'+base_name)
        cropped_image = img.crop((coordinates[0],coordinates[1],coordinates[2],coordinates[3]))

        return cropped_image

    # 获取文本区域的坐标信息
    def get_coordinates(self, img, image_name, boxes, scale):

    	# 获取需要检测的图片名称
        base_name = image_name.split('/')[-1]
        # to save detected text area's coordinates
        min_y_sort_list = []
        for box in boxes:
            if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                continue
            min_x = min(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            min_y = min(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            max_x = max(int(box[0]/scale),int(box[2]/scale),int(box[4]/scale),int(box[6]/scale))
            max_y = max(int(box[1]/scale),int(box[3]/scale),int(box[5]/scale),int(box[7]/scale))
            line = [min_x, min_y, max_x, max_y]
            min_y_sort_list.append(line)
            # to sort coordinates' y 
        min_y_sort_list = self.sort_list(min_y_sort_list)

        return min_y_sort_list, base_name

    # ctpn检测文本区域
    def ctpn(self, image_name):
        img = cv2.imread(image_name)
        img, scale = self.resize_im(img, scale=600, max_scale=1000) # 参考ctpn论文
        scores, boxes = test_ctpn(self.sess, self.net, img)
        # ctpn识别实例
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        min_y_sort_list, base_name = self.get_coordinates(img, image_name, boxes, scale)

        return min_y_sort_list, base_name

    # crnn文本信息识别
    def crnn_recognition(self, cropped_image):

        converter = utils.strLabelConverter(self.alphabet)
      
        image = cropped_image.convert('L')
        w = int(image.size[0] / (image.size[1] * 1.0 / 32))
        transformer = dataset.resizeNormalize((w, 32))
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        self.model.eval()
        preds = self.model(image)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        characters_prediction = converter.decode(preds.data, preds_size.data, raw=False)
        return characters_prediction

    def endtoend_det_rec_init(self):

        if torch.cuda.is_available():
            model = self.model.cuda()
        print('loading pretrained model from %s' % self.crnn_model_path)
        # 导入已经训练好的crnn模型
        model.load_state_dict(torch.load(self.crnn_model_path))
        try:
            # 导入已训练好的模型
            ckpt = tf.train.get_checkpoint_state(self.ctpn_model_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        except:
            raise '导入失败, 请检查路径是否正确!'
        img_names = os.listdir(self.detect_data_path)
        images_name_list = []
        for img_name in img_names:
            img_name = os.path.join(self.detect_data_path,img_name)
            images_name_list.append(img_name)

        return images_name_list
        

if __name__ == '__main__':
    
    timer = Timer()
    timer.tic()
    # init
    example = det_and_rec()
    #
    images_name_list = example.endtoend_det_rec_init()
    with open(example.txt_results_path, 'w') as txt_reco:
        for image_name in images_name_list:
            txt_reco.write(image_name + '\n')
            print('recognizing..[%s]' %(image_name))
            min_y_sort_list, base_name = example.ctpn(image_name)
            for coordinates in min_y_sort_list:
                cropped_image =example.crop_images(coordinates, base_name)
                characters_prediction = example.crnn_recognition(cropped_image)
                txt_reco.write(characters_prediction + '\n')
    timer.toc()
    print(timer.total_time)