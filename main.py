import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from argparse import ArgumentParser
import datetime as dt

def load_images(index):
    images_return = []
    path = './imgs/'
    # dapi - Celulas sem marcação, gfp - Ceulas marcadas como morta e união das duas
    img_name = np.array(['20170809_3D LD DAY 5_Area1_#_DAPI.tif', '20170809_3D LD DAY 5_Area1_#_GFP.tif'])
    for item in img_name:
        images_return.append(cv2.cvtColor(cv2.imread(path + item.replace('#', str(index))), cv2.COLOR_BGR2RGB)) 
    
    return images_return

def log_transformation(constant, image):
    '''
    source: https://medium.com/@sonu008/image-enhancement-contrast-stretching-using-opencv-python-6ad61f6f171c
    
    Can be used to brighten the intensities of an image (like the Gamma Transformation, where gamma < 1).
    More often, it is used to increase the detail (or contrast) of lower intensity values.
    '''
    
    return constant * (np.log(1 + np.float32(image)))

def kmeans(image):
    shape_2 = 1 if len(image.shape) == 2 else image.shape[2]
    reshaped = image.reshape(image.shape[0] * image.shape[1], shape_2)
    kmeans = KMeans(n_clusters=2, n_init=40, max_iter=500).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8), (image.shape[0], image.shape[1]))
    return clustering

def contrast_stretching(img, E):
    '''
    source: http://www.cs.uregina.ca/Links/class-info/425/Lab3/
    
    Contrast-stretching transformations increase the contrast between the darks and the lights.
    E controls the slope of the function and M is the mid-line where you want 
    to switch from dark values to light values, we use the the mean of the image intensities.
    np.spacing(1) is aconstant that is the distance between 1.0 and the next largest number 
    that can be represented in double-precision floating point.
    
    '''

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _mean = img.reshape((img.shape[0]*img.shape[1], 1)).mean(axis=0)
    M = _mean
    cs = 1 / ( 1 + (M / (img + np.spacing(1))) ** E)
    return np.float32(cs)

def pcs(img, tx = 0.1):
    '''
    source: http://interscience.in/IJPPT_Vol2Iss1/29-34.pdf
    
    Partial  contrast  is  an  auto  scaling  method.  It  is  a  
    linear   mapping   function   that   is   usually   used   to   
    increase the contrast level and brightness level of the 
    image.  This  technique  will  be  based  on  the  original  
    brightness  and  contrast  level  of  the  images  to  do  the  
    adjustment.  
    
    r_max,  b_max  and  g_max  are  the  maximum  
    color level for each red, blue and green color palettes, 
    respectively. r_min, b_min and g_min are the 
    minimum  value  for  each  color  palette,  respectively.  
    maxTH  and  minTH  are  the  average  number  of  these  
    maximum  and  minimum  color  levels  for  each  color  
    space. 
    
    '''
    r, g, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    r_min = r.reshape((r.shape[0]*r.shape[1], 1)).min(axis=0)
    g_min = g.reshape((g.shape[0]*g.shape[1], 1)).min(axis=0)
    b_min = b.reshape((b.shape[0]*b.shape[1], 1)).min(axis=0)
    r_max = r.reshape((r.shape[0]*r.shape[1], 1)).max(axis=0)
    g_max = g.reshape((g.shape[0]*g.shape[1], 1)).max(axis=0)
    b_max = b.reshape((b.shape[0]*b.shape[1], 1)).max(axis=0)
    minTH = (r_min + g_min + b_min)/3
    maxTH = (r_max + g_max + b_max)/3
    NminTH = minTH - minTH * tx
    NminTH = 0 if NminTH < 0 else NminTH
    NmaxTH = maxTH + (255 - maxTH) * tx
    NmaxTH = 255 if NmaxTH > 255 else NmaxTH
    
    #print('minTH {}, Nmin {}, maxTH {}, Nmax {}'.format(minTH, NminTH, maxTH, NmaxTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    _min = img.reshape((img.shape[0]*img.shape[1], 1)).min(axis=0)
    output = np.zeros_like(img)
    
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            
            if img[x, y] > minTH :
                output[x,y] = (img[x,y]/minTH) * NminTH
            elif minTH <= img[x, y] and img[x, y] <= maxTH :
                output[x,y] = (((NmaxTH - NminTH)/(maxTH - minTH)) * (img[x,y] - _min)) + _min
            elif img[x, y] < maxTH :
                output[x,y] = (img[x,y]/maxTH) * maxTH
                
    return output

def find_microenvironments(img2print, mask, area_min = 3000, count_cnt_min = 100):    
    mask = mask.copy()
    img2print = img2print.copy()
    _mask = np.zeros_like(img2print)
    microenvironments = []

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < area_min:
            continue

        if len(cnt) < count_cnt_min:
            continue
        '''
        if len(microenvironments) > 0 :
            continue
        '''

        
        _e = list(cv2.fitEllipse(cnt))
        
        #excluindo raios muito grandes
        if(_e[1][0] > 200 or _e[1][1] > 200):
            continue
        
        #aumentando os raios da elipse para exluir algum erro.
        #print(area, len(cnt), _e[1])
        _e[1] = (_e[1][0] + 50, _e[1][1] + 50)
        ellipse = tuple(_e) 
        
        mask = np.zeros_like(img2print)
        cv2.ellipse(mask, ellipse, (255,255,255), -1)
        masked = np.bitwise_and(img2print, mask)
        microenvironments.append(masked)
        cv2.ellipse(img2print, ellipse, (0,255,0), 4)
        cv2.putText(img2print,str(len(microenvironments) - 1), (int(ellipse[0][0]), int(ellipse[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.ellipse(_mask, ellipse, (255,255,255), -1)
        
    
    return (microenvironments, img2print, _mask)


def counter_me(me, ix):
    row = ['{}'.format(ix), '-','-','-','-','-','-','-','-','-','-']
    for i, m in enumerate(me):

        _masked = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(_masked,5)
        ret,thresh1 = cv2.threshold(blur,20,255,cv2.THRESH_BINARY) #THRESH_TOZERO     THRESH_BINARY
        _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cells = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)

            if(cnt.shape[0] < 5):
                continue

            ellipse = cv2.fitEllipse(cnt)  
            #cv2.ellipse(img2print, ellipse, (0,255,0), 4)
            cells += 1

        row[i+1] = cells
        #print('{} Microambiente {}; {} células mortas encontradas'.format(k, i, cells))
        
    return row
    
    
def save_img(img, mask, ix):    
   cv2.imwrite('./outputs/out_{}.png'.format(ix),img)
   cv2.imwrite('./outputs/out_mask_{}.png'.format(ix),mask)

def process(index):
    
    ini = dt.datetime.now()
    indexs = [index]
    if(index == '0'):
        indexs = range(1, 64)

    
    df = pd.DataFrame(columns = ['MICROAMBIENTE', 'me 0', 'me 1', 'me 2','me 3', 'me 4','me 5', 'me 6','me 7', 'me 8','me 9'])
    for ix in indexs:
        print(ix)
        images = load_images(ix)
        cs = pcs(images[0], 0.1)
        logtrans = log_transformation(0.9, cs)
        blur = cv2.medianBlur(logtrans,5)
        kmask = kmeans(blur)        
        me, img, mask = find_microenvironments(images[1], kmask)
        save_img(img,kmask, ix)
        df.loc[len(df)] = counter_me(me, ix)

    df.to_csv('./outputs/output.csv')
    print('Tempo de execução {} '.format((dt.datetime.now()-ini)))



parser = ArgumentParser()
parser.add_argument('-i', dest='index')

args = parser.parse_args()
process(args.index)


