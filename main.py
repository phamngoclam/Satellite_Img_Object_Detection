# This script shows the full training and prediction pipeline for a pixel-based classifier:
# we create a mask, train logistic regression on one-pixel patches, make prediction for all pixels, create and smooth polygons from pixels.


from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

############################################
import matplotlib.pyplot
import matplotlib.pyplot as plt
from matplotlib import colors
import copy
from time import time
import pandas as pd
import os
###########################################

csv.field_size_limit(sys.maxsize);

NUM_CLASS = 7
#TRAIN_SET = ['6120_2_2', '6120_2_0','6140_3_1','6100_2_3','6140_1_2','6110_3_1']
#TEST_SET = ['6120_2_2','6110_4_0', '6110_3_1']

TRAIN_SET = ['6040_2_2','6120_2_2','6120_2_0','6090_2_0','6040_1_3']

#TEST_SET = ['6120_2_0','6110_3_1']
#TEST_SET = ['6160_2_1', '6170_0_4', '6170_4_1'] #nhieu 5 it 6 nen mau xanh
#TEST_SET = ['6040_1_0', '6010_4_4'] #10_6_23_2
TEST_SET = ['6100_2_3', '6100_2_2'] #10_7_23_2_5_6 sub_lay_2_5_6
#TEST_SET = ['6100_2_2'] #10_8_24_1_8_5_1
#TEST_SET = ['6110_4_0', '6120_2_0', '6100_2_3'] #'6110_2_3 hinh moi' sub_4_9_10


#LIST_CLASS = ['1','5','8']
#LIST_CLASS = ['1','2','3','4','5','6','7','8','9','10']
LIST_CLASS = ['1','3','5','6','7','8']
BAND_WIDTH = 20

inDir = '../input'
result_file = '../input/predict/20band/test/'
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
cmap = colors.ListedColormap([ '#FFFFFF', '#B2001F','#F19373','#B7B7B7','#FEF889','#367517', '#AFD788', '#00B2BF', '#00676B', '#211551', '#8273B0'])
# mau do: nha, mau kem: cong trinh dan dung, mau xam nhat: duong, mau vang nhat: duong nho, mau xanh dam: cay, mau xanh nhat: tham co
# mau xanh duong: duong thuy, mau xanh duong dam: vung nuoc, mau tim dam: xe tai lon, mau tim nhat: xe tai nho
poly_type=[0 ,1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
norm = colors.BoundaryNorm(poly_type, cmap.N)

# Load grid size
def get_xmax_ymin(IM_ID):
    x_max = y_min = None
    for _im_id, _x, _y in csv.reader(open('../input/grid_sizes.csv')):
        if _im_id == IM_ID:
            x_max, y_min = float(_x), float(_y)
            break
    return x_max, y_min

def get_size_tiff(im_rgb):
    im_size = im_rgb.shape[:2]
    return im_size

# Scale polygons to match image:

def get_scalers(im_size, x_max, y_min):
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    w_ = w * (w / (w + 1.))
    h_ = h * (h / (h + 1.))
    return w_ / x_max, h_ / y_min

def get_train_polygons_scaled(train_polygons, im_size, x_max, y_min):
    x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
    train_polygons_scaled = shapely.affinity.scale(train_polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    return train_polygons_scaled

# Load train poly with shapely
def get_train_polygons(IM_ID, type):
    train_polygons = None
    for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):
        if _im_id == IM_ID and _poly_type == type:
            train_polygons = shapely.wkt.loads(_poly)
            break
    return train_polygons


def get_mask(IM_ID, im_size):
    x_max, y_min = get_xmax_ymin(IM_ID)
    x_scaler, y_scaler = get_scalers(im_size, x_max, y_min)
    train_mask = np.zeros(im_size, np.uint8)
    for _im_id, _poly_type, _poly in csv.reader(open('../input/train_wkt_v4.csv')):
        if ((_im_id == IM_ID) and (_poly != 'MULTIPOLYGON EMPTY') and (_poly_type in LIST_CLASS)):
            print("-------type------", _poly_type)
            polygon = get_train_polygons(_im_id, _poly_type)
            polygon_scaled = get_train_polygons_scaled(polygon, im_size, x_max, y_min)
            train_mask = get_train_mask(train_mask, polygon_scaled, im_size, int(_poly_type))
    return train_mask

def imread_RGB(IM_ID, ismask):
    img_rgb = tiff.imread('../input/three_band/{}.tif'.format(IM_ID)).transpose([1, 2, 0])
    im_size = get_size_tiff(img_rgb)
    if ismask==True:
        train_mask = get_mask(IM_ID, im_size)
    else:
        train_mask = None
    return img_rgb, train_mask

def imread_sixtenband(IM_ID, _type, ismask):
    if _type == 'P':
        img = tiff.imread('../input/sixteen_band/{}.tif'.format(IM_ID+'_'+_type))
        im_size = img.shape

        if ismask == True:
            train_mask = get_mask(IM_ID, im_size)
        else:
            train_mask = None
    else:        
        img = tiff.imread('../input/sixteen_band/{}.tif'.format(IM_ID+'_'+_type)).transpose([1, 2, 0])
        im_size = get_size_tiff(img)

        if ismask == True:
            train_mask = get_mask(IM_ID, im_size)
        else:
            train_mask = None

    return img, train_mask

def crop_img(img, size):
    img_tmp = img[0:size, 0:size, :]
    return img_tmp

def crop_mask(mask, size):
    mask_tmp = mask[0:size, 0:size]
    return mask_tmp

def get_im_rgb20(IM_ID):
    img_M, train_mask = imread_sixtenband(IM_ID, 'M', True)

    size = min(get_size_tiff(img_M))
    img_M = crop_img(img_M, size)
    x, y = get_size_tiff(img_M)

    img_M_resize = cv2.resize(img_M, (x, y)) 
    tmpM = np.rollaxis(img_M_resize, 2, 0)
    #showTiff(tmpM, IM_ID)

    img_rgb, mask_rgb = imread_RGB(IM_ID, False)
    size = min(get_size_tiff(img_rgb))
    img_rgb = crop_img(img_rgb, size)
    img_rgb_resize = cv2.resize(img_rgb, (x, y)) 
    #showTiff(img_rgb_resize, IM_ID)

    img_A, mask_A = imread_sixtenband(IM_ID, 'A', False)
    size = min(get_size_tiff(img_A))
    img_A = crop_img(img_A, size)
    img_A_resize = cv2.resize(img_A, (x, y)) 
    tmpA = np.rollaxis(img_A_resize, 2, 0)
    #showTiff(tmpA,IM_ID)

    img_P, mask_P = imread_sixtenband(IM_ID, 'P', False)
    size = min(img_P.shape)
    img_P = crop_mask(img_P, size)
    img_P_resize = cv2.resize(img_P, (x, y)) 

    Array = np.zeros((img_rgb_resize.shape[0],img_rgb_resize.shape[1],20))
    Array[..., 0:3] = img_rgb_resize
    Array[..., 3:11] = img_M_resize
    Array[..., 11:19] = img_A_resize
    Array[..., 19] = img_P_resize

    size = min(train_mask.shape)
    train_mask = crop_mask(train_mask, size)

    return Array, train_mask

def mask_for_polygons(img_mask, polygons, im_size, _type):
    if not polygons:
        return img_mask

    tmp_mask = np.zeros(img_mask.shape, np.uint8)
    tmp_mask = copy.copy(img_mask)
    int_coords = lambda x: np.array(x).round().astype(np.int32)  # ep kieu sang kieu int
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons] # Lam tron so
    interiors = [int_coords(pi.coords) for poly in polygons             #https://toblerity.org/shapely/manual.html
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, _type)
    cv2.fillPoly(img_mask, interiors, 0)

    row, col = img_mask.shape
    for x in xrange(0,row):
        for y in xrange(0,col):
            if tmp_mask[x,y] != 0 and tmp_mask[x,y] != _type:
                img_mask[x,y] = tmp_mask[x,y]

    return img_mask

def get_train_mask(img_mask, train_polygons_scaled, im_size, _type):
    train_mask = mask_for_polygons(img_mask, train_polygons_scaled, im_size, _type)
    return train_mask

# A helper for nicer display
def scale_percentile(matrix):
    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)
    # Get 2nd and 98th percentile
    mins = np.percentile(matrix, 1, axis=0)
    maxs = np.percentile(matrix, 99, axis=0) - mins
    matrix = (matrix - mins[None, :]) / maxs[None, :]
    matrix = np.reshape(matrix, [w, h, d])
    matrix = matrix.clip(0, 1)
    return matrix

# Check that image and mask are aligned. Image:
def showTiff(im_rgb, title):
    #fig = plt.figure()
    tiff.imshow(255 * scale_percentile(im_rgb[:,:]));
    plt.title(title)
    plt.tight_layout()
    #plt.show()
    plt.savefig(result_file+title+'.png')
    plt.close()

#And mask:
def show_mask(m):
    tiff.imshow(255 * np.stack([m, m, m]));
    plt.show()

def show_train_mask():
    show_mask(train_mask[:,:])
    plt.title("train mask")
    plt.show()

def show_module(data):
    show_mask(data[:,:])
    plt.show()

def plot_matrix(rm, title='Figure', cmap=plt.cm.Blues, norm  = norm):
    fig = plt.figure()
    plt.imshow(rm, interpolation='nearest', cmap=cmap, norm = norm)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    # fig.savefig(result_file+title+'.png')
    # plt.close(fig)

def show_mask_color(matrix, title):
    plot_matrix(matrix, title, cmap=cmap, norm = norm)

def find_repres(x, y, im_rgb, train_mask, size):

    count_class = np.zeros(NUM_CLASS, int)

    sub_rgb = np.array((size, size, BAND_WIDTH))
    sub_rgb = im_rgb[x : x + size,y : y + size,:]

    sub_mask = np.array((size, size))
    sub_mask = train_mask[x : x + size, y : y + size]

    row_mask, col_mask = sub_mask.shape

    for row in xrange(0, row_mask):
        for col in xrange(0, col_mask):
            count_class[sub_mask[row, col]] = count_class[sub_mask[row, col]] + 1

    class_repres = np.argmax(count_class)
    color_repres = np.zeros((BAND_WIDTH), int)
    count = 0

    for row in xrange(0, row_mask):
        for col in xrange(0, col_mask):
            if sub_mask[row, col] == class_repres:
                count += 1
                color_repres += sub_rgb[row, col, :]

    color_repres /= count

    return color_repres, class_repres

def scale(im_rgb, train_mask, size):
    x_rgb, y_rgb, z_rgb = im_rgb.shape

    row = x_rgb % size
    col = y_rgb % size

    if (row == 0):
        row = x_rgb / size
    else:
        row = x_rgb / size + 1

    if (col == 0):
        col = y_rgb / size
    else:
        col = y_rgb / size + 1

    new_rgb = np.zeros((row, col, BAND_WIDTH))
    new_mask = np.zeros((row, col))

    row_mask, col_mask = -1, -1

    for x in range(0,x_rgb, size):
        row_mask += 1
        for y in range(0,y_rgb, size):
            col_mask += 1
            color_repres, class_repres = find_repres(x, y, im_rgb, train_mask, size)

            new_mask[row_mask, col_mask] = class_repres
            new_rgb[row_mask, col_mask, :] = color_repres

        col_mask = -1

    return new_rgb, new_mask


# Now, let's train a very simple logistic regression classifier,
# just to get some noisy prediction to show how output mask is processed.


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import average_precision_score

from sklearn import linear_model

def get_xs_ys(im_rgb, train_mask):
    xs = im_rgb.reshape(-1,BAND_WIDTH).astype(np.float32)
    ys = train_mask.reshape(-1)
    return xs, ys

def apply_pipeline_new_image(IM_ID, _type, pipeline, jaccard_list):
    if _type != 'RGB' and _type != '20' :
        img, train_mask = imread_sixtenband(IM_ID, _type, True)
    elif _type == '20':
        img,train_mask = get_im_rgb20(IM_ID)
    elif _type == 'RGB':
        img, train_mask = imread_RGB(IM_ID, True)

    print("show origin",IM_ID,"_____image shape =", img.shape)
    img_rgb, mask = imread_RGB(IM_ID, False)
    showTiff(img_rgb, IM_ID)
    show_mask_color(train_mask, IM_ID + " -a origin test")
    print("train_mask origin", train_mask)
    xs, ys = get_xs_ys(img, train_mask)

    return jaccard_index(pipeline, xs, train_mask, IM_ID, jaccard_list)


def caculate_jacard(jaccard_list, IM_ID, predict, result):
    for i in range(1,NUM_CLASS):
        pre = (predict == i)
        res = (result == i)
        tp, fp, fn = (( pre &  res).sum(),
              ( pre & ~res).sum(),
              (~pre &  res).sum())

        jaccard_list[i,0] += jaccard_list[i,0] + tp
        jaccard_list[i,1] += jaccard_list[i,1] + fp
        jaccard_list[i,2] += jaccard_list[i,2] + fn

    print("pred_binary_mask &  train_mask",predict &  result)
    tp, fp, fn = (( predict &  result).sum(),
                  ( predict & ~result).sum(),
                  (~predict &  result).sum())
    print('type pred_binary_mask',type(predict))
    print ('tp',tp)
    print ('fp',fp)
    print ('fn',fn)
    if fp == 0. and tp == 0. and fn == 0. :
        jaccard_index = 0.
    else:
        jaccard_index = float(tp +0.001) / (tp + fp + fn)
    show_mask_color(predict, IM_ID+" - image predict - "+str(jaccard_index))
    print('Pixel jaccard', jaccard_index)
    print("**************************************************")

    return jaccard_list

def jaccard_index(pipeline, xs, train_mask,IM_ID, jaccard_list):

    #pred_ys = pipeline.predict_proba(xs)
    #print("--------- pipeline coef_", pipeline.coef_)
    #print("predict ys ", pred_ys)
    pred_result = pipeline.predict(xs).astype(np.int32)
    print("shape result predict ", pred_result.shape)
    print("predict result", pred_result)

    # xresult, yresult = pred_ys.shape
    # print("xresult ",xresult)
    # count_type = np.zeros((NUM_CLASS), int)
    # for x in range(0,xresult):
    #     count_type[pred_result[x]] += 1
    # print("+++++++++++++++++++++++++++++++++++++++++ count_type ", count_type)

    pred_result = pred_result.reshape(train_mask.shape)

    print("predict type result", pred_result)
    print("shape predict result", pred_result.shape)
    print("train_mask shape", train_mask.shape)

    print("********** jaccard of image",IM_ID," *********")
    jaccard_list = caculate_jacard(jaccard_list, IM_ID, pred_result, train_mask)
        
    return jaccard_list


def predict_id(IM_ID, model):
    img,train_mask = get_im_rgb20(IM_ID)
    xs, ys = get_xs_ys(img, train_mask)
    size = get_size_tiff(img)
    pred_result = model.predict(xs).astype(np.int32)
    pred_result = pred_result.reshape(size)

    return pred_result
    

def predict_test(IM_ID, model):
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        if id == IM_ID:
            msk = predict_id(id, model)
            np.save('../msk1/10_%s' % id, msk)
    

def get_rgb_mask(im_rgb, train_mask, xs, ys):
    rgb = xs.reshape(im_rgb.shape).astype(np.int32)
    mask = ys.reshape(train_mask.shape).astype(np.int32)

    return rgb, mask


def mask_to_polygons(mask, epsilon=5, min_area=1.):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(
        ((mask == 1) * 255).astype(np.uint8),
        cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def make_submit():
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    _img_old =  ''
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1]
        msk = np.load('../msk1/10_%s.npy' % id)
        msk1 = msk == kls
        pred_polygons = mask_to_polygons(msk1)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)
        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                  origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        print "===== make id =", id, " type =", kls
        if idx % 100 == 0: print idx
    print df.head()
    df.to_csv('../subm/2.csv', index=False)


            #jaccard_list = apply_pipeline_new_image(id,'20', pipeline, jaccard_list)


def get_xset_yset():
    #pipeline = linear_model.SGDClassifier(loss='log', random_state = 0,warm_start = True, n_jobs = -1, n_iter = 15)
    #pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log', n_jobs = -1, warm_start = True, n_iter = 15))
    x_all = np.empty((0,BAND_WIDTH), int)
    y_all = np.array([], dtype = int)
    _break = 0
    _img_old = ''
    for _im_id_all, _poly_type_all, _poly_all in csv.reader(open('../input/train_wkt_v4.csv')):
        if ((_im_id_all != 'ImageId') and (_poly_all != 'MULTIPOLYGON EMPTY') and (_im_id_all != _img_old)):

            _img_old = _im_id_all
            _break = _break + 1

            print("========== image ", _break,"_____ ID =",_im_id_all,"===================")

            img, train_mask = get_im_rgb20(_im_id_all)

            print("shape image", img.shape)
            print('shape train mask', train_mask.shape)

            xs, ys = get_xs_ys(img, train_mask)
            im_tmp = np.rollaxis(img, 2, 0)


            #rgb, mask = get_rgb_mask(im_rgb, train_mask, xs, ys)
            #print("shape rgb ", rgb.shape)
            #showTiff(im_tmp, _im_id_all)
            #rgb_scale, mask_scale = scale(img, train_mask, 5)
            #showTiff(rgb_scale, _im_id_all)
            #print("shape rgb_scale ", rgb.shape)
            #xs_new, ys_new = get_xs_ys(rgb_scale, mask_scale)

            x_all = np.append(x_all, xs, axis = 0)
            y_all = np.append(y_all, ys, axis = 0)

            print("x_all shape new", x_all.shape)
            print("y_all shape new", y_all.shape)

            #show_mask_color(train_mask, "image origin - " + _im_id_all)
            #show_mask_color(mask_scale, "image scale - " + _im_id_all)
            # pipeline.fit(xs, ys, coef_init = coef)

            if _break >= 25:
                print("x_all", x_all)
                print("y_all", y_all)
                print("shape x_all",x_all.shape[:])
                print("shape y_all",y_all.shape[:])
                break
    print('training...')
    pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='modified_huber',warm_start=True,n_jobs = -1))
    print("pipeline",pipeline)
    pipeline.fit(x_all,y_all)

    print("==========**********==========")
    #print("coef ",pipeline.coef_)
    print("==========**********==========")
    return pipeline

if __name__ == '__main__':
    t0 = time()
    jaccard_list = np.zeros((NUM_CLASS,3), float)
    model = get_xset_yset()
    t1 = time()
    
    # for img in TEST_SET:
    #     print('----------test',img,'----------')
    #     jaccard_list = apply_pipeline_new_image(img,'20', model, jaccard_list)

    
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    _img_old =  ''
    dem = 0
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1
        if _img_old != id :
            dem += 1
            print("   test :",dem," ---- ",id)
            _img_old = id
            predict_test(id, model)
            #jaccard_list = apply_pipeline_new_image(id,'20', pipeline, jaccard_list)

    make_submit()

    # print("time train", t1 - t0)
    # print("time predict", t2 - t1)
    

    # _img_old =  ''
    # for _im_id_all, _poly_type_all, _poly_all in csv.reader(open('../input/train_wkt_v4.csv')):
    #     if ((_im_id_all != 'ImageId') and (_poly_all != 'MULTIPOLYGON EMPTY') and (_im_id_all != _img_old) and (_im_id_all not in TRAIN_SET)):
    #         _img_old =  _im_id_all
    #         jaccard_list = apply_pipeline_new_image(_im_id_all,'20', pipeline, jaccard_list)

    t2 = time()
    print("time train   :", t1 - t0)
    print("time predict :", t2 - t1)
    
    jaccard_index = 0.
    for x in range(1, NUM_CLASS):
        tp = jaccard_list[x,0]
        fp = jaccard_list[x,1]
        fn = jaccard_list[x,2]

        if (tp != 0. or fp != 0. or fn !=0.):
            jaccard_index += float(tp +0.001) / (tp + fp + fn)
        elif (tp == 0 and fp == 0 and fn == 0):
            jaccard_index += 0.

    print("length of train", len(TRAIN_SET))
    print("image train", TRAIN_SET)

    print("length of test", len(TEST_SET))
    print("image test", TEST_SET)

    print('Jaccard final', jaccard_index/float(NUM_CLASS - 1))
    tp = sum(jaccard_list[:,0])/float(NUM_CLASS - 1)
    fp = sum(jaccard_list[:,1])/float(NUM_CLASS - 1)
    fn = sum(jaccard_list[:,2])/float(NUM_CLASS - 1)

    print('Jaccard final 2', float(tp +0.001) / (tp + fp + fn))


    file = open(result_file +'result.txt', 'w')

    file.write("   time train    :"+ str(t1 - t0) +'\n')
    file.write("   time predict  :"+ str(t2 - t1) +'\n')
    file.write("   length of train 22 \n")
    file.write("   image train not"+ str(TEST_SET) +'\n')

    file.write("   length of test :" + str(len(TEST_SET))+'\n')
    file.write("   image test   :" + str(TEST_SET))

    file.write('   Jaccard avarage'+ str(float(tp +0.001) / (tp + fp + fn)) +'\n')
    file.write('   Jaccard final'+ str(jaccard_index/float(NUM_CLASS - 1)) +'\n')
    file.close()
