# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 17:09:44 2023

@author: liubo
"""
import cv2
import requests
import json
import numpy as np
import pandas as pd
import os
import glob
from scipy.linalg import det
from scipy.spatial.distance import euclidean
from sklearn.linear_model import LinearRegression
# %%
# original file path
d = glob.glob("examples\\ori\\*.jpg")


# %%
# python3_cv2读取路径含中文
# https://blog.csdn.net/qq_38846606/article/details/115525390
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T)), np.matrix([0., 0., 1.])])

def dist2d(a, b, c):
    v1 = b - c
    v2 = a - b
    m = np.column_stack((v1, v2))
    d = abs(np.linalg.det(m)) / np.sqrt(np.sum(v1 * v1))
    return d


# Function to calculate the angle between three points
def angle2d(A, B, C):
    AB = B - A
    AC = C - A
    norm_AB = np.sqrt(np.sum(AB ** 2))
    norm_AC = np.sqrt(np.sum(AC ** 2))
    dot_product = np.sum(AB * AC)
    cosine_angle = dot_product / (norm_AB * norm_AC)
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# Function to find the perpendicular point from P to line AB
def find_perpendicular_point(P, A, B):
    x0, y0 = P
    x1, y1 = A
    x2, y2 = B
    if x1 == x2:
        x_perp = x1
        y_perp = y0
    else:
        k = (y2 - y1) / (x2 - x1)
        k_perp = -1 / k
        A = k_perp
        B = -1
        C = y0 - k_perp * x0
        a = -k
        b = 1
        c = k * x1 - y1
        Delta_x = B * c - b * C
        Delta_y = A * c - a * C
        Det = A * b - a * B
        x_perp = Delta_x / Det
        y_perp = Delta_y / Det
    return np.array([x_perp, y_perp])


# Function to calculate the angle between two lines
def calculate_angle_between_lines(A1, B1, A2, B2):
    vec1 = B1 - A1
    vec2 = B2 - A2
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the lines has zero length.")
    dot_product = np.dot(vec1, vec2)
    cos_angle = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg if angle_deg <= 180 else 360 - angle_deg


# resize pics
out_path = "examples\\resize\\"
for f in d:
    img = cv_imread(f)
    vis = img.copy()
    vis = cv2.resize(vis, (int(vis.shape[1] * 0.5), int(vis.shape[0] * 0.5)))
    cv2.imencode('.jpg', vis)[1].tofile(out_path + f.split("\\")[-1])  # python3_cv2保存路径含中文
    # cv2.imwrite(out_path+f.split("\\")[-1],vis) #python2可能可行

# %%
# 106 points
url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
payload = {'api_key': 'jm499WpdnMzlXeUeKRBuQuAUryJvyBxN',
           'api_secret': 'EuKoxYYPgXBIiSMGrjhQsekKCmpuFT5O',
           'return_landmark': 2,
           'return_attributes': 'headpose,eyestatus,mouthstatus'}
out_path = "examples\\points\\"
resize_pics = glob.glob("examples\\resize\\*.jpg")
resize_pics.sort()


fo = open("examples/resize/check.txt", "w", encoding="UTF-8")
fo.write('filename\tface_num\twidth\theight\troll_angle\n')
for f in resize_pics:
    print(f)
    files = {'image_file': open(f, 'rb')}
    r = requests.post(url, files=files, data=payload)
    print(r)
    data = json.loads(r.text)
    print(data['face_num'])
    if data['face_num'] == 1:
        width = data['faces'][0]['face_rectangle']['width']
        top = data['faces'][0]['face_rectangle']['top']
        height = data['faces'][0]['face_rectangle']['height']
        left = data['faces'][0]['face_rectangle']['left']
        fo.write(
            f + '\t' + str(data['face_num']) + '\t' + str(data['faces'][0]['face_rectangle']['width']) + '\t' + str(
                data['faces'][0]['face_rectangle']['height']) + '\t' + str(
                data['faces'][0]['attributes']['headpose']['roll_angle']) + '\n')
        img = cv2.imread(f)
        vis = img.copy()
        cv2.rectangle(vis, (left, top), (left + width, top + height), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX

        ##extract landmark
        m = 0
        for j in range(0, len(data['faces'])):
            values = []
            dtype = [('Point', '|U32'), ('X', float), ('Y', float)]
            for i in data['faces'][j]['landmark']:
                cor = data['faces'][j]['landmark'][i]
                x = cor["x"]
                y = cor["y"]
                values.append((list(data['faces'][0]['landmark'].keys())[m], x, y))
                m = m + 1
            lmk = np.array(values, dtype=dtype)
            lmk = np.sort(lmk, order='Point')
            for cont in range(0, len(lmk)):
                cv2.circle(vis, (int(lmk[cont][1]), int(lmk[cont][2])), 5, (0, 255, 0), -1)
                # cv2.putText(vis,str(cont),(int(lmk[cont][1]),int(lmk[cont][2])),font,0.5,(255,255,255),1,False)
            np.savetxt(out_path + f.split("\\")[-1] + ".csv", lmk, delimiter=",", fmt="%s")
            m = 0

        cv2.imwrite(out_path + f.split("\\")[-1], vis)


    else:
        fo.write(f + '\t0\t0\t0\t360\n')
fo.close()

#%% GPA 
avg=np.matrix(np.genfromtxt('support_files\\point106avgface_all.txt',delimiter='\t'))
files=glob.glob(os.path.join('examples\\points\\','*.csv'))
res=[]
filenames=[]
for f in files:
    points1=np.matrix(np.genfromtxt(f,delimiter=','))
    points1=points1[:,[1,2]]
    M1=transformation_from_points(points1,avg)
    filename=os.path.basename(f)
    filenames.append(filename)
    points2=np.matrix(np.genfromtxt(os.path.join('examples\\points\\',filename),delimiter=','))
    points2=points2[:,[1,2]]
    Landmarks2=points2
    L1=np.matrix([[point[0,0], point[0,1],1] for point in Landmarks2] )
    L1=np.transpose(L1)
    P1=M1*L1
    P1=np.transpose(P1)
    Q1=np.matrix([[p[0,0],p[0,1]] for p in P1])
    res.append(np.array(Q1.ravel())[0])
    # 确保目标目录存在，如果不存在则创建
    output_dir = os.path.join('examples', 'points', 'GPA')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # 创建目标目录

    # 保存文件
    np.savetxt(os.path.join(output_dir, filename), Q1, delimiter=',')
res=np.array(res)

#%% calculate_facial_feature
# 读取文件并处理数据
files = glob.glob("examples/points/GPA/*.csv")
sample_res_all = pd.DataFrame()

for i in range(0,len(files)):
    sample_res_all = pd.DataFrame(None)
    data = pd.read_csv(files[i], header=None)
    sample_id = os.path.splitext(os.path.basename(files[i]))[0]

    sample_res = pd.DataFrame(0, index=[0], columns=["ID"])
    sample_res["ID"] = sample_id

    miankuan = euclidean(data.iloc[13], data.iloc[29])
    sample_res["face_width"] = miankuan

    miangao = euclidean(data.iloc[72], data.iloc[0])
    sample_res["face_height"] = miangao

    midline_points = data.iloc[[72, 80, 53, 0]]
    model = LinearRegression().fit(midline_points[[0]], midline_points[[1]])
    midline_p1_x = np.ceil(min(midline_points[0]))
    midline_p1_y = model.predict([[midline_p1_x]])[0][0]
    midline_p2_x = np.ceil(max(midline_points[0]))
    midline_p2_y = model.predict([[midline_p2_x]])[0][0]
    midline_p1 = np.array([midline_p1_x, midline_p1_y])
    midline_p2 = np.array([midline_p2_x, midline_p2_y])

    # 计算各种特征
    p4 = data.iloc[3].values
    p4_midline = dist2d(p4, midline_p1, midline_p2)
    p20 = data.iloc[19].values
    p20_midline = dist2d(p20, midline_p1, midline_p2)
    sample_res["HP0009940_1"] = p4_midline / p20_midline

    p17 = data.iloc[16].values
    p17_midline = dist2d(p17, midline_p1, midline_p2)
    p33 = data.iloc[32].values
    p33_midline = dist2d(p33, midline_p1, midline_p2)
    sample_res["HP0009940_2"] = p17_midline / p33_midline

    sample_res["HP0012801_02_1"] = euclidean(p4, p20)
    sample_res["HP0012801_02_2"] = euclidean(p17, p33)

    p6 = data.iloc[5].values
    p15 = data.iloc[14].values
    p22 = data.iloc[21].values
    p31 = data.iloc[30].values
    sample_res["HP0005446_1"] = (angle2d(p4, p17, p6) + angle2d(p20, p33, p22)) / 2
    sample_res["HP0005446_2"] = (angle2d(p17, p15, p4) + angle2d(p33, p31, p20)) / 2

    p1 = data.iloc[0].values
    p54 = data.iloc[53].values
    sample_res["HP0000331_0400000"] = euclidean(p1, p54)

    p8 = data.iloc[7].values
    p24 = data.iloc[23].values
    sample_res["HP0011822_1"] = euclidean(p6, p22)
    sample_res["HP0011822_2"] = euclidean(p8, p24)

    sample_res["HP0000307"] = angle2d(p1, p6, p22)

    p2 = data.iloc[1].values
    p18 = data.iloc[17].values
    sample_res["HP0025386"] = euclidean(p2, p18)

    p53 = data.iloc[52].values
    p62 = data.iloc[61].values
    sample_res["HP0000154_0000160"] = euclidean(p53, p62)

    p53_midline = dist2d(p53, midline_p1, midline_p2)
    p62_midline = dist2d(p62, midline_p1, midline_p2)
    sample_res["HP0009941_1"] = p53_midline / p62_midline

    p65 = data.iloc[64].values
    p65_midline = dist2d(p65, midline_p1, midline_p2)
    p69 = data.iloc[68].values
    p69_midline = dist2d(p69, midline_p1, midline_p2)
    sample_res["HP0009941_2"] = p65_midline / p69_midline

    p64 = data.iloc[63].values
    p64_midline = dist2d(p64, midline_p1, midline_p2)
    p68 = data.iloc[67].values
    p68_midline = dist2d(p68, midline_p1, midline_p2)
    sample_res["HP0009941_3"] = p64_midline / p68_midline

    p56 = data.iloc[55].values
    p56_midline = dist2d(p56, midline_p1, midline_p2)
    p59 = data.iloc[58].values
    p59_midline = dist2d(p59, midline_p1, midline_p2)
    sample_res["HP0009941_4"] = p56_midline / p59_midline

    p57 = data.iloc[56].values
    p57_midline = dist2d(p57, midline_p1, midline_p2)
    p60 = data.iloc[59].values
    p60_midline = dist2d(p60, midline_p1, midline_p2)
    sample_res["HP0009941_5"] = p57_midline / p60_midline

    p36 = data.iloc[35].values
    p40 = data.iloc[39].values
    p90 = data.iloc[89].values
    p94 = data.iloc[93].values
    sample_res["eye_width_avg"] = (euclidean(p36, p40) + euclidean(p90, p94)) / 2

    p78 = data.iloc[77].values
    p84 = data.iloc[83].values
    sample_res["nose_width"] = euclidean(p78, p84)

    p72 = data.iloc[71].values
    p63 = data.iloc[62].values
    sample_res["HP0000215_0000219"] = euclidean(p72, p63)

    p61 = data.iloc[60].values
    sample_res["HP0000179_0010282"] = euclidean(p61, p54)

    sample_res["HP0000188_0011341"] = euclidean(p65, p69)

    if data.iloc[71, 1] > data.iloc[63, 1] and data.iloc[71, 1] > data.iloc[67, 1]:
        sample_res["HP0002263_0010804_0010806"] = angle2d(p72, p64, p68)
    else:
        sample_res["HP0002263_0010804_0010806"] = -angle2d(p72, p64, p68)

    sample_res["HP0000316_0000601"] = euclidean(p40, p90)

    p36 = data.iloc[35].values
    p94 = data.iloc[93].values
    sample_res["HP0000316_0000601_out"] = euclidean(p36, p94)

    p73 = data.iloc[72].values
    p87 = data.iloc[86].values
    sample_res["HP0003194_0033142"] = euclidean(p73, p87)

    p80 = data.iloc[79].values
    p86 = data.iloc[85].values
    sample_res["HP0012810"] = euclidean(p80, p86)

    p87_midline = dist2d(p87, midline_p1, midline_p2)
    sample_res["HP0011831_1"] = p87_midline
    sample_res["HP0011831_2"] = euclidean(p87, p78) / euclidean(p87, p84)

    sample_res["HP0000289_0011829"] = euclidean(p64, p68)

    p81 = data.iloc[80].values
    sample_res["HP0000343_0000322"] = euclidean(p81, p72)

    p81_64_midline_angle = calculate_angle_between_lines(p81, p64, midline_p1, midline_p2)
    if p81_64_midline_angle > 90:
        sample_res["HP0011827"] = 180 - p81_64_midline_angle
    else:
        sample_res["HP0011827"] = p81_64_midline_angle

    sample_res_all = pd.concat([sample_res_all, sample_res], ignore_index=True)

    # 移除不需要的列
    ncol_remove = [0, 3, 5, 7, 8, 11, 13, 16, 17, 18, 19, 30]
    colnames_remove = ["Face_Width", "Face_Height", "Jaw_Symmetry", "Jaw_Width", "Chin_Height", "Chin_Width",
                       "Chin_Pointed", "Mouth_Width", "Mouth_Symmetry", "Eye_Width_avg", "Nose_Width", "Upperlip_Thickness",
                       "Lowerlip_Thickness", "Upperlip_Width", "Upperlip_Shape", "Eye_Distance_inner", "Eye_Distance_outer",
                       "Nasalbridge_Height", "Nasalbase_Width", "Nasaltip_Deviation", "Philtrum_Width", "Philtrum_Height",
                       "Philtralridge"]
    pheno_remove = None
    pheno_remove = sample_res_all.drop(sample_res_all.columns[ncol_remove], axis=1)
    pheno_remove.columns = colnames_remove
    
    # 读取参考数据并计算Z-score
    reference = pd.read_csv("support_files/HPO_meansd.txt", sep="\t")
    # 让 reference 具有与 pheno_remove 相同的索引
    reference.index = (pheno_remove.iloc[0]).index
    
    pheno_remove.loc[1] = (pheno_remove.iloc[0] - reference["All_mean"]) / reference["All_sd"]
    pheno_remove.index = ["value", "Zscore"]
    
    # 保存结果
    pheno_remove.to_csv(sample_id+"_result.txt", sep="\t", index=True, header=True, quoting =False)
