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
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--detect_images_path', type=str, default='detect_data/detect_images', help='the path to your images')
opt = parser.parse_args()

# ctpn params
ctpn_model_path= 'trained_models/trained_ctpn_models'
save_coordinates_path = 'coordinates_results/'
detect_data_path = opt.detect_images_path
cropped_images_path = 'detect_data/cropped_images/'

# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'trained_models/trained_crnn_models/mixed_1p5m_second_finetune_acc97p7.pth'
txt_results_path = 'detect_data/txt_results/text_info_results.txt'
alphabet = str1
nclass = len(alphabet)+1

# 图片resize
def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])

    return cv2.resize(im, 
    	None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

# 对获得的坐标由上到下排序
def sort_list(min_y_sort_list):
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
def crop_images(coordinates, base_name, num_of_boxes, model, img):
    global txt_reco
    txt_reco.write(base_name+'\n')
    img_2 = Image.open(detect_data_path+'/'+base_name)
    '''
    test cropped images
    '''
    # new_file = './test_cropped_images/'+base_name+'/'
    # if os.path.exists(new_file):
    #     shutil.rmtree(new_file)
    # os.makedirs(new_file)

    for i in range(num_of_boxes):
        cropped_image = img_2.crop((coordinates[i][0]-8 if coordinates[i][0]!=0 else coordinates[i][0],
                                    coordinates[i][1]-1.5,coordinates[i][2]+8,coordinates[i][3]-0.5))
        # cropped_image = img_2.crop((coordinates[i][0],
        #                     coordinates[i][1],coordinates[i][2],coordinates[i][3]))
        crnn_recognition(cropped_image, model)

        '''
        test cropped images
        '''
        # cropped_image.save(new_file+str(i)+".png")

# 获取文本区域的坐标信息
def get_coordinates(img, image_name, boxes, scale, model):
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
    min_y_sort_list = sort_list(min_y_sort_list)
    crop_images(min_y_sort_list, base_name, len(min_y_sort_list), model, img)

# to binary
def image_to_binary(img):
    r, g, b = cv2.split(img)
    a = np.ones(img.shape[:2],dtype="uint8") * 255
    r = a - r
    r[r>=5] += 255

    return r

# ctpn检测文本区域
def ctpn(sess, net, image_name, model):
    img = cv2.imread(image_name)

    #r = image_to_binary(img)
    #noise = np.ones(img.shape[:2],dtype="uint8") * 125
    #img = cv2.merge((r+noise, r, noise))
    
    img, scale = resize_im(img, scale=600, max_scale=1000) # 参考ctpn论文
    print('ctpn', img.shape)
    scores, boxes = test_ctpn(sess, net, img)
    # ctpn识别实例
    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    get_coordinates(img, image_name, boxes, scale, model)

# crnn文本信息识别
def crnn_recognition(cropped_image, model):

    global txt_reco
    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    txt_reco.write(sim_pred+'\n')

def endtoend_det_rec():
    global txt_reco
    global detect_data_path
    global txt_results_path
    # tensorflow GPU内存分配0.75,按需分配
    config=tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    sess = tf.Session(config=config)
    # load network
    net = get_network("VGGnet_test")
    # load model
    saver = tf.train.Saver()
    # crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from %s' % crnn_model_path)
    # 导入已经训练好的crnn模型
    model.load_state_dict(torch.load(crnn_model_path))
    try:
        # 导入已训练好的模型
        ckpt = tf.train.get_checkpoint_state(ctpn_model_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    except:
        raise 'import error, please check the path!'
    # focus on png or jpg
    print(glob.glob(os.path.join(detect_data_path, '*.png')))
    img_names = glob.glob(os.path.join(detect_data_path, '*.png')) + \
               glob.glob(os.path.join(detect_data_path, '*.jpg'))
    # txt_reco 保存识别的文本信息
    txt_reco = open(txt_results_path, 'w')
    for img_name in img_names:
        print('Recognizing...[{0}]'.format(img_name))
        ctpn(sess, net, img_name, model)
    txt_reco.close()

if __name__ == '__main__':
    
    timer = Timer()
    timer.tic()
    endtoend_det_rec()
    timer.toc()
    print(timer.total_time)
    