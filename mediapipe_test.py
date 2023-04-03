import cv2
import mediapipe as mp
import os
import numpy as np
import re
import matplotlib.pyplot as plt 
import math

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

from scipy.spatial import Delaunay
from sklearn.decomposition import PCA

from sklearn.metrics.pairwise import rbf_kernel



IMAGE_FILES = [os.path.join("./test_imgs", p) for p in os.listdir("./test_imgs")]

RIGHT_IRIS = [(469, 470), (470, 471), (471, 472),(472, 469)]
LEFT_IRIS = [(474, 475), (475, 476), (476, 477),(477, 474)]

LEFT_EYEBROW = [(276, 283), (283, 282), (282, 295),(295, 285),  # lower eyebrow
                (300, 293), (293, 334),(334, 296), (296, 336) # up eyebrow
                ]

RIGHT_EYEBROW = [(46, 53), (53, 52), (52, 65), (65, 55),  # lower eyebrow
                (70, 63), (63, 105), (105, 66), (66, 107) # up eyebrow
                ]

LIPS = [(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
        (17, 314), (314, 405), (405, 321), (321, 375),(375, 291),  # down lip
        
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
        (267, 269), (269, 270), (270, 409), (409, 291),  # up lip 
        
        (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
        (14, 317), (317, 402), (402, 318), (318, 324),
        (324, 308),   # inner up lip
        
        (78, 191), (191, 80), (80, 81), (81, 82),
        (82, 13), (13, 312), (312, 311), (311, 310),
        (310, 415), (415, 308) # inner down lip 
        ] 

LEFT_EYE = [(263, 249), (249, 390), (390, 373), (373, 374),
            (374, 380), (380, 381), (381, 382), (382, 362), # lower eye
            
            (263, 466), (466, 388), (388, 387), (387, 386),
            (386, 385), (385, 384), (384, 398), (398, 362) # up eye
            ]

RIGHT_EYE = [(33, 7), (7, 163), (163, 144), (144, 145),
            (145, 153), (153, 154), (154, 155), (155, 133), # lower eye
            (33, 246), (246, 161), (161, 160), (160, 159),
            (159, 158), (158, 157), (157, 173), (173, 133) # up eye
            ]

OVAL = [(10, 338), (338, 297), (297, 332), (332, 284),
        (284, 251), (251, 389), (389, 356), (356, 454),
        (454, 323), (323, 361), (361, 288), (288, 397),
        (397, 365), (365, 379), (379, 378), (378, 400),
        (400, 377), (377, 152), (152, 148), (148, 176),
        (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234),
        (234, 127), (127, 162), (162, 21), (21, 54),
        (54, 103), (103, 67), (67, 109), (109, 10)] # clockwise from top to top

NOSE = [(114, 188), (188, 122), (122, 6), (6, 351), (351, 412), (412, 343),  # up first line 
        (437, 399), (399, 419), (419, 197), (197, 196), (196, 174), (174, 217), # second line
        (198, 236), (236, 3), (3, 195), (195, 248), (248, 456), (456, 420), # third line 
        (360, 363), (363, 281), (281, 5), (5, 51), (51, 134), (134, 131), # fourth line 
        (115, 220), (220, 45), (45, 4), (4, 275), (275, 440),(440, 344), (344, 278), # fifh line
        (294, 439), (439, 438), (438, 457), (457, 274), (274, 1), (1, 44), (44, 237), (237, 218), (218, 219), # six line
        (235, 75), (75, 60), (60, 20), (20, 242), (242, 141), (141, 94), (94, 370), (370, 462), (462, 250), (250, 290), # seven
        (305, 455), (455, 439) # eight
        ]
NOSETRIL = [(79, 20), (20, 60), (60, 166), (166, 79), # right
            (309, 392), (392, 290), (290, 250), (250, 309) # left
            ]

ALL_LINE = RIGHT_IRIS + LEFT_IRIS + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS + LEFT_EYE + RIGHT_EYE + OVAL + NOSE + NOSETRIL

from scipy.spatial import ConvexHull

# 可分成两个类，一个解析人脸，一个映射到骨骼参数

    
def resize_and_pad_image(image, size):
    """
    将图片按比例resize到1024*1024(size)，差的地方做pad处理
    """
    # 获取原图尺寸
    h, w = image.shape[:2]
    # 确定缩放比例
    scale = min(size[0]/w, size[1]/h)
    # 计算缩放后的尺寸
    new_size = (int(w*scale), int(h*scale))
    # 缩放图像
    resized_image = cv2.resize(image, new_size)
    # 计算需要填充的像素数
    pad_w = size[0] - new_size[0]
    pad_h = size[1] - new_size[1]
    # 计算填充的边界大小
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    # 填充图像
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return padded_image

class MeshPoints2BoneParameters(object):
    def __init__(self, data, **keargs) -> None:
        """
        landmarks3d478
        all is left -> right , up -> lower, outer -> inner
        """
        # self.landmarks3d478 = data  # 先做实验，后确定写法
        self.data = data
        self.eyebrow_index = ([276, 283, 282, 295, 285], [300, 293, 334, 296, 336], 
                           [46, 53, 52, 65, 55], [70, 63, 105, 66, 107])
        self.eye_index = ([263, 466, 388, 387, 386, 385, 384, 398, 362],[263, 249, 390, 373, 374, 380, 381, 382, 362], 
                          [33, 246, 161, 160, 159, 158, 157, 173, 133], [33, 7, 163, 144, 145, 153,154, 155, 133])
        self.eye_index_mid = ([386, 374], [159, 145])
        self.iris_index = ([474, 475, 476, 477], [469, 470, 471, 472])  # 左右
        self.nose_index = ([114, 188, 122, 6, 351, 412, 343, 437, 399, 419, 197, 196, 174, 217, 198, 236, 3, 195, 248, 
                            456, 420, 360, 363, 281, 5, 51, 134, 131],)
        self.nose_hole_index = ([309, 392, 290, 250], [79, 20, 60, 166])
        self.lips_index = ([61, 146, 91, 181, 84,17, 314, 405, 321, 375, 291], [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
                           [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308], [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291])
        self.cheek_index = ([])
        self.nose_height_index = ([6, 1, 94],)  # 94 最下层
        self.nose_line_index = ([114, 188, 122, 6, 351, 412, 343], [437, 399, 419, 197, 196, 174, 217], 
                                [198, 236, 3, 195, 248, 456, 420])
        self.outer_edge_index = ([])
        # self.eye_dis = self.distance()
        self.character_size = ()
        self.input_face_size = ()
        # eye shape 
        self.iris_oval_dist_index = ([53], [103, 67])  # 右眼眉毛到人脸边缘的垂直距离~到边缘两点距离的平均值
        self.left_eye_corner_index_l = [466, 263, 249]  # 左眼左眼角
        self.left_eye_corner_index_r = [398, 362, 382]  # 左眼右眼角
        self.left_eye_contour_index = list(set(self.eye_index[0] + self.eye_index[1]))  # 计算眼睛面积
        # self.face_info = dict({"眉毛角度":0, "眉毛高度":0})
        self.fp_names = ["眉毛角度", "眉毛高度", "眼睛位置", "眼睛宽度", "眼球大小", "眼球位置", "眼睛样式",
                         "眼尾", "瞳距", "眼睛突出", "鼻子位置", "鼻子大小", "鼻梁宽度", "鼻尖样式", "鼻子高度",
                         "嘴巴位置", "嘴巴宽度", "嘴巴样式", "嘴巴突出", "下巴长度", "下巴样式", "下巴厚度",
                         "下巴尖样式", "下巴突出", "下巴角度", "脸部宽度", "颧骨样式", "脸颊"]
        self.results_index_fp_names = list(range(len(self.fp_names)))
        self.fp_name_en = ["Eyebrow angle", "Eyebrow height", "Eye position", "Eye width", "Eyeball size", "Eyeball position", "Eye style",
                        "Eye end", "Pupil distance", "Eye prominence", "Nose position", "Nose size", "Nasal bridge width", "Nose tip style",
                        "Nose height", "Mouth position", "Mouth width", "Mouth style", "Mouth prominence", "Chin length", "Chin style",
                        "Chin thickness", "Chin tip style", "Chin prominence", "Chin angle", "Facial width", "Zygomatic style", "Cheek"
                        ]
        iris_coords_left = self.data[self.iris_index[0], :2]
        self.iris_center_left = (iris_coords_left[:, 0].mean(), iris_coords_left[:, 1].mean())
        # 计算距离的量均需要除以相对尺度
        self.base_eye_iris_d = 5.0  # 貌似难以解决， 压缩尝试
        self.base_wh = 256
        
        # 实际传入参数，还需要调试对应参数。。。
        
    def area_poly(self, points):
        # Calculate the convex hull of the points
        hull = ConvexHull(points)
        # Reorder the points based on the convex hull
        
        ordered_points = points[hull.vertices]
        # Calculate the area of the polygon
        n = ordered_points.shape[0]
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += ordered_points[i][0] * ordered_points[j][1]
            area -= ordered_points[i][1] * ordered_points[j][0]
        area = abs(area) / 2
        return area
    
    def traingle_area(self, points):
        A, B, C = points
        # Calculate the area of the triangle
        BA = A - B
        BC = C - B
        area = 0.5 * np.abs(np.cross(BA, BC))
        return area
    
    def slope(self, p1, p2):
        if p1[0] == p2[0]:
            # 如果两点的x坐标相等，斜率无穷大
            return float('inf')
        else:
            return (p2[1] - p1[1]) / (p2[0] - p1[0])
        
    def angle(self, p1, p2):
        
        radian = math.atan(self.slope(p1, p2))
        return math.degrees(radian), radian
    
    def theta(self, points):
        a, b, c = points
        ab = a - b
        bc = c - b
        theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))
        # Convert the angle from radians to degrees
        theta_degrees = theta * 180 / np.pi
        
        return theta_degrees
    
    def eyeybrow_shape(self):
        """
        2
        眉毛角度：-1， 1， [靠近鼻子部分曲线点，斜率从-k变到0,或0.j]
        眉毛高度：-1， 3， [与瞳孔的距离，与上眼睛线与瞳孔在竖直方向上的交点的距离, 下眉毛和上眼围着的面积]
        优化1：眉毛线用上下线的中线来替换
        优化2：考虑角度和高度的关联作用
        """
        p1, p2, p3, p4, p5 = self.data[self.eyebrow_index[1], :2]
        # lineup = self.data[self.eyebrow_index[0], :2]
        # linedown = self.data[self.eyebrow_index[1], :2]
        # p1, p2, p3, p4, p5  = (lineup + linedown ) / 2 # line_mid, p1,-> p5 从左到右的顺序. 用靠近鼻子的右边两点
        # iris_coords = self.data[self.iris_index[0], :2]
        # iris_center = (iris_coords[:, 0].mean(), iris_coords[:, 1].mean())
        iris_eyebrow_d = self.distance(self.iris_center_left, p3)
        dx = self.distance(self.data[336, :2], p3)  # 296, 336
        
        # eb1, eb2 = self.data[[285, 282], :2]
        # ebd1, ebd2 = self.data[[336, 334], :2]
        eb_m1, eb_m2 = (self.data[[285, 283], :2] + self.data[[336, 293], :2]) / 2  # 四个点首末,三个点角度要大一点
        # sk = self.slope(eb1, eb2)
        # sk = self.slope(p3, p5)
        # sk = self.angle(eb1, eb2)
        sk, radian = self.angle(eb_m1, eb_m2)
        dy = dx * radian * 2
        iris_eyebrow_d = iris_eyebrow_d - dy
        
        # 标准图片 4.5~15
        
        return [sk, iris_eyebrow_d]
    
    def sigmoid(self, x):
        """
        压缩输出值与参考值的差异，开放图片预测的值范围更大。 无法和参考值做有效比较
        """
        return 1/(1+np.exp(-x))
    
    def tanh(self, x):
        """
        差异与参考值的0值作比较
        """
        return (1 - np.exp(-2*x)) / (1 + np.exp(2*x))
    
    def eye_openrate_iris_shift(self):
        """
        8
        眼睛在中间位置上眼皮和下眼皮的距离，瞳孔距离上下眼皮的距离，
        用瞳孔和上下眼皮的交的比例，来确定眼睛闭合的程度。因为瞳孔的四个点的顺序没有
        确认，只好取两对角线的最大值，将open_rate 和 眼球位置联系
        
        位置：上下移动， -1， 1 [眉毛与上边缘的距离]
        宽度：大小，左右眼角长度， -1， 1 [面积大小，或左右两边角的夹角大小]
        眼球大小： 变化不明显，随机， -1， 1 [随机]
        眼球位置：上下翻动，-1， 1， （-1向下，0最大中间，1最上）[上下曲线的距离，和瞳孔矩形与上下曲线的交的程度，上1]
        眼睛样式：大小，-1， 0.5， 边角左边水平-1， 大小缩小，且左边弧度上翘0.5[下曲线两边的弯曲度]
        眼尾：-1， 1， [和眼睛样式差不多，区别在于眼角的厚度，眼角部分黑色宽度，-1，大，1很小]
        瞳距：-1, 1 [瞳孔中心距离，眉毛内距离]
        眼睛突出，-1, 1 [随机]
        
        优化1.位置变为，眼球与上边缘的距离, 考虑到眉毛的参数改变，其中心位置也跟着变带来的二次修正考虑
        优化2. 眼尾， 眼角高度与瞳孔高度的偏差
        优化3. 眼睛宽度，眼睛两圈选定点集合的的距离平均值
        优化4. 眼睛样式， 眼角和瞳孔中心的竖直偏移 不行。 计算左眼角部分的曲率半径，曲率半径越大，0.5, 越小，-1
        """
        # results = []
        # 眼睛位置
        # l_eye_i, up_iris_m = self.iris_oval_dist_index[0], self.iris_oval_dist_index[1]
        # eye_locat = (self.distance(self.data[l_eye_i[0], :2], self.data[up_iris_m[0], :2]) + 
        #              self.distance(self.data[l_eye_i[0], :2], self.data[up_iris_m[1], :2]))/2
        # 62.5 63.2, diff 0.7
        eye_locat = self.distance(self.data[297,:2], self.iris_center_left)
        # 宽度 eye_area命名不好
        # 可以根据数据的统计特征或者其他特征来确定当前值的更小范围，采样
        # eye_area = self.area_poly(self.data[self.left_eye_contour_index, :2])  # 当面积很小，以及open_rate小时，取值为0
        # 4.3, 4.9 diff 0.6
        eye_area = (self.distance(self.data[263, :2], self.data[359, :2]) + \
            self.distance(self.data[249, :2], self.data[255, :2]) + \
            self.distance(self.data[390, :2], self.data[339, :2]) + \
            self.distance(self.data[466, :2], self.data[467, :2]) + \
            self.distance(self.data[388, :2], self.data[260, :2]) ) / 5
        # 眼球大小
        scale = np.random.rand(1) * 2 - 1

        # 眼球位置
        # open_rate 0-1， -1, 1. *2 - 1
        # iris_coords = self.data[self.iris_index[0], :2]
        p0, p1, p2, p3 = self.data[self.iris_index[0], :2]
        pu, pl = self.data[self.eye_index_mid[0][0], :2], self.data[self.eye_index_mid[0][1], :2]
        # iris_center = (iris_coords[:, 0].mean(), iris_coords[:, 1].mean())
        iris_lenght = max(self.distance(p0, p2), self.distance(p1, p3))
        eye_open_dist = self.distance(pu, pl)
        iris_pu_d = self.distance(pu, self.iris_center_left)
        iris_pl_d = self.distance(pl, self.iris_center_left)
        eye_open_lenght = iris_pu_d + iris_pl_d
        open_rate = min(eye_open_dist, eye_open_lenght) / iris_lenght
        # 眼睛样式 小0.5， 大-1
        # eye_corner = (self.theta(self.data[self.left_eye_corner_index_l, :2]) + 
        #               self.theta(self.data[self.left_eye_corner_index_r, :2]) ) / 2
        # eye_corner = self.data[359, 1] - self.iris_center_left[1]
        # 计算曲率半径
        # 0.5 26.9, -1 30.7 diff 3.8 
        avg_curvature, mid_curvature = self.bezier_curve_curvature(self.data[[374, 373, 390, 249], :2])
        # print(avg_curvature, mid_curvature, 'avg mid')
        eye_corner = 1/avg_curvature
        # 眼尾 高1, 低-1, >0, 1, <0, -1
        eye_tail = (self.data[263, 1] + self.data[362, 1]) / 2 - self.iris_center_left[1]
        #瞳距
        # -1 77.5, 1 89 diff 11.5
        p1 = np.mean(self.data[self.iris_index[0], :2], axis=0)
        p2 = np.mean(self.data[self.iris_index[1], :2], axis=0)
        iris_dist = self.distance(p1, p2)
        # 眼睛突出
        eye_t = np.random.rand(1) * 2 - 1
        
        return [eye_locat, eye_area, scale[0], open_rate, eye_corner, eye_tail, iris_dist, eye_t[0]]
    
    def statics_paramerter(self, img_list):
        """
        在给定各种数据上统计出各个数据的分布，数据应该在尺度，图片大小，表情上均要比较丰富 TODO 
        """
        pass
    
    def nose_shape(self):
        """
        5
        鼻子位置：-1， 1, [鼻孔大小，鼻子顶部中点和鼻子底部中点，在z轴上的夹角，鼻梁长度]
        鼻子大小：-1， 1， [鼻孔大小，鼻孔处左右宽度] 位置和大小，两者有合成效果，鼻孔最大是两者都位置-1，大小1
        鼻梁宽度：-0.5, 1， [鼻梁线曲率，中间线条在x轴上的左右距离平均值]
        鼻尖样式：-1， 1， [鼻孔处斜率，鼻尖到孔边缘的斜率，负->正]
        鼻子高度：-0.5,1， [随机，或者鼻梁中间线的z值大小]
        
        优化1. 
        """
        n_hole_area = self.area_poly(self.data[self.nose_hole_index[0], :2])
        p1, p2, p3 = self.data[self.nose_height_index[0], 1:]
        n_angle = self.theta((p1, p2, p3)) # yz 平面
        np_l1 = self.data[[188, 412], :2]  # p1, p2 self.nose_line_index[0][1,5]
        np_l2 = self.data[[399, 174], :2]  # self.nose_line_index[1][1,5]
        np_l3 = self.data[[236, 456], :2]  # 
        n_width = sum(map(lambda x: abs(x[0][0] - x[1][0]), [np_l1, np_l2, np_l3]))/3
        p1, p2 = self.data[self.nose_hole_index[0][1:3], :2]
        n_pick_style = self.slope(p1, p2)
        n_heigt = (1 - np.random.rand(1) * 1.5 ) * p2[1]
        
        return [n_hole_area, n_angle, n_width, n_pick_style, n_heigt[0] ]
    
    def mouth_shape(self):
        """
        4
        嘴巴位置：-1， 1， [与鼻子的距离]
        嘴巴宽度：-1.5， 1 [嘴角两边长度]
        嘴巴样式：-1， 1 [嘴角两边斜率， 负->0]
        嘴巴突出：-1， 1 [嘴巴中心z值，或随机]
        """
        m_locate = self.distance(self.data[94, :2], self.data[0, :2])
        m_width = self.distance(self.data[61, :2], self.data[308, :2])
        m_style_r = self.theta(self.data[[40, 61, 91], :2])
        m_style_l = self.theta(self.data[[270, 291, 321], :2])
        m_style = (m_style_r + m_style_l) / 2
        m_convex = (self.data[13, 2] + self.data[14, 2]) / 2
        return [m_locate, m_width, m_style, m_convex]
    
    def jaw_face_hipbone(self):
        """
        8
        下巴长度：0,1 [下巴底部与嘴唇的距离]
        下巴样式：-1, 1 [腮帮的曲率,或平均z值]
        下巴厚度：-1， 1 []  # TODO
        下巴尖样式：-0.5, 1, [下巴形状，尖->宽]
        下巴突出：-0.5， 1， [下巴z值，或随机]
        下巴角度：-1, 1, [和下巴样式一样]
        脸部宽度：-1, 1, [脸的宽度]
        髋骨样式：-1, 1 [髋骨部分曲面凸出， 凹陷->凸出， 也许平均z值, 顶部到四周的点的z轴距离]
        脸颊：-1， 1， [腮帮部分曲线的凸出率]
        """
        # jaw_index = [176, 148, 152, 377,  400]
        hipbone_index = [342, 353, 265, 261, 255, 359, 466]  # 466, 中间点
        jaw_length = self.distance(self.data[152, :2], self.data[17, :2])
        jaw_style = np.mean(self.data[[367, 397, 288, 435, 401, 361], 2])
        jaw_thickness =  np.random.rand(1) * 2 - 1
        jaw_pick_style = self.theta(self.data[[176, 152, 400], :2])
        p2 = self.data[152, 2]
        jaw_convex = (1 - np.random.rand(1) * 1.5 ) * p2
        jaw_theta = jaw_style
        face_width = self.distance(self.data[234, :2], self.data[454, :2])
        # 髋骨样式, 小于0， 则凸， 大于0， 则凹
        hipbone_style = np.mean(self.data[hipbone_index[:-1], 2]) - self.data[hipbone_index[-1], 2]
        # 脸颊
        # xs = self.data[[367, 397, 288, 435, 401, 361], 0]
        # ys = self.data[[367, 397, 288, 435, 401, 361], 1]
        # # print(xs, ys)  # xs 存在重复
        # ps, po = self.spline_curve(xs, ys)
        # jaw_c = self.cacl_curvature(ps)
        # face_j = jaw_c
        face_j = 0
        return [jaw_length, jaw_style, jaw_thickness[0], jaw_pick_style, jaw_convex[0], jaw_theta, face_width, hipbone_style, face_j]
        
    def transformer_all_parameters(self, r):
        # transformer brow
        # 效果均不好，说明这两者不是简单的线性关系。
        # B = np.array([[20.031, 14.323, 6.009, 5.768, 15.005, 4.764, 12.653, 13.761], 
        #               [14.336, 27.226, 25.502, 33.088, 22.025, 26.416, 17.893, 24.499]])
        # C = np.array([[-1, -1, 1, 1, -1, 1, 0, 0], [-1, 3, -1, 3, 0, 0, -1, 3]])
        # B = np.array([[20.031, 14.323, 6.009, 5.768], 
        #               [14.336, 27.226, 25.502, 33.088]])
        # C = np.array([[-1, -1, 1, 1], [-1, 3, -1, 3]])
        # A = self.solve_two_relative(B, C)
        # print("矩阵A的结果为：\n", A)
        # r_b = r[0]
        # r_h = r[1]
        r_b, r_h = r[:2]
        if 5 < r_b < 20:
            # r[0] = r[0] / (50) * 2 - 1
            # 此处是非线性关系
            if r_b <= 6:
                r[0] = 1
            elif 11 < r_b < 13:
                r[0] = 0
            elif r_b >= 14:
                r[0] = -1
            else:
                r[0] = np.exp(6-r_b) # 6-11 1/exp(x-6)
        elif r_b >= 20:
            r[0] = -1
        elif abs(r_b) <= 5:
            # 不考虑头部偏角太大，否则负角度太大,没做头都变换
            r[0] = 1
        else:
            r[0] = 0
            
        # transformer eye
        eye_locat, eye_area, scale, open_rate, eye_corner, eye_tail, iris_dist, eye_t = r[2:10]
        # 没做统计分析之前，处理意义不大
        
        return r
    
    def solve_two_relative(self, B, C):
        """
        两个控制量a, b. 分别有各自范围， 两者存在组合关系，范围边缘四个，各自为0时四个。共计8个
        或者简单的范围边缘四个
        B.shape = (2, 8)
        C.shape = (2, 8)
        A.shape = (2, 2)
        B.T * A.T = C.T
        A * B = C
        """
        A = np.linalg.lstsq(B.T, C.T, rcond=None)[0].T
        return A
    
    def test_all_handle(self):
        results = []
        results += self.eyeybrow_shape()
        results += self.eye_openrate_iris_shift()
        results += self.nose_shape()
        results += self.mouth_shape()
        results += self.jaw_face_hipbone()
        return results
        
    def distance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    def bezier_curve(self, t, P0, P1, P2, P3):
        return (1-t)**3*P0 + 3*t*(1-t)**2*P1 + 3*t**2*(1-t)*P2 + t**3*P3
    
    def bezier_curve_curvature(self, points):
        """
        # P0 = np.array([1, 1])
        # P1 = np.array([2, 3])
        # P2 = np.array([4, 3])
        # P3 = np.array([5, 2])
        points: [np.array([x, y], ..., np.array([x, y])]
        """
        P0, P1, P2, P3 = points
        # 生成nx个等间隔的t值
        ts = np.linspace(0, 1, 20)
        # 计算曲线上每个点的坐标
        points = np.array([self.bezier_curve(t, P0, P1, P2, P3) for t in ts])

        # 计算曲线的导数和二阶导数
        avg_curvature, mid_curvature = self.cacl_curvature(points)
        # for test
        # self.draw_curve(points, save_n='bcc')
        
        return avg_curvature, mid_curvature
    
    def draw_curve(self, points, original_points=None, save_n='bc'):
        # Plot the original points and the fitted curve
        # plt.plot(x, y, 'o', label='Original Points')
        xs = points[:, 0]
        ys = points[:, 1]
        plt.plot(xs, ys, label='Spline Curve')
        if original_points is not None:
            x = original_points[:, 0]
            y = original_points[:, 1]
            plt.plot(x, y, 'o', label='Original Points')
        plt.xlim([0, 1024])
        plt.ylim([0, 1024])
        plt.legend()
        # plt.show()
        plt.savefig(f'{save_n}.png')
    
    def cacl_curvature(self, curve_points):
        # 计算曲线的导数和二阶导数
        first_deriv = np.gradient(curve_points, axis=0)
        second_deriv = np.gradient(first_deriv, axis=0)

        # 计算曲率
        curvature = np.abs(np.cross(first_deriv, second_deriv)) / np.power(np.sum(np.power(first_deriv, 2), axis=1), 1.5)

        # 计算平均曲率
        avg_curvature = np.mean(curvature)
        mid_curvature = curvature[len(curvature) // 2]
        # print("The average curvature of the Bezier curve is:", avg_curvature)
        return avg_curvature, mid_curvature
    
    def test_mouth(self, x):
        pass
    
    def spline_curve(self, x, y, save_n='b'):
        """
        is right 
        x = [1, 3, 4, 6]
        y = [3, 4, 2, 5]
        """
        from scipy.interpolate import make_interp_spline
        # Define the coordinates of four points
        sorted_index = np.array(x).argsort()
        x = x[sorted_index]
        y = y[sorted_index]
        # 得出的坐标存在值相同，导致make_interp_spline无法调用
        # np.where(x[:-1 !=x[1:]])[0]

        # Generate a spline curve from the four points
        curve = make_interp_spline(x, y)

        # Evaluate the spline curve at a set of x coordinates
        x_new = np.linspace(min(x), max(x), 20)
        y_new = curve(x_new)
        points = np.concatenate((x_new.reshape(-1, 1), y_new.reshape(-1, 1)), axis=1)
        original_points = np.concatenate((x.reshape(-1,1), y.reshape(-1, 1)), axis=1)
        # c = self.cacl_curvature(points)
        # print("ccccc", c)
        # g = self.cacl_curve_mean(curve, x_new)
        # print("ggggg", g)

        self.draw_curve(points, original_points, save_n=save_n)
        return points, original_points
        
    def test_3d_simulation(self, x, y, z, n_grid=20):
        xi, yi, zi = self.rbf_interpolation(x, y, z, n_grid=n_grid)
        K = self.compute_gaussian_curvature(zi)
        c = self.is_concave(K)
        print(c)
        self.draw_3dcurve(x, y, z, xi, yi, zi, K)
        
    def cacl_curve_mean(self, curve, x_new):
        """
        curve : curve = make_interp_spline(x, y), 这个和self.cacl_curvature算出来的不同 TODO 
        """
        # Calculate the curvature of the spline curve at each point
        dydx = curve(x_new, 1)
        d2ydx2 = curve(x_new, 2)
        curvature = np.abs(d2ydx2) / (1 + dydx ** 2) ** 1.5
        mean_curvature = np.mean(curvature)
        
        return mean_curvature
    
    def rbf_interpolation(self, x, y, z, n_grid=10):
        """
        # 生成一些随机点
        np.random.seed(0)
        n_points = 100
        x = np.random.rand(n_points) * 10
        y = np.random.rand(n_points) * 10
        z = np.sin(np.sqrt(x**2 + y**2)) + np.random.normal(scale=0.1, size=n_points)
        """
        # 构造 3D 网格点
        xi = np.linspace(x.min(), x.max(), n_grid)
        yi = np.linspace(y.min(), y.max(), n_grid)
        xi, yi = np.meshgrid(xi, yi)
        xi_flat = xi.flatten()
        yi_flat = yi.flatten()

        # 计算核矩阵
        X = np.column_stack([x, y])
        K = rbf_kernel(X)

        # 求解线性方程组
        w = np.linalg.solve(K, z)

        # 计算插值值
        X_test = np.column_stack([xi_flat, yi_flat])
        K_test = rbf_kernel(X, X_test)
        zi = np.dot(K_test.T, w).reshape(xi.shape)
        
        return xi, yi, zi
    
    def compute_gaussian_curvature(self, surface):
        """
        G is the Gaussian curvature and H is the mean curvature,
        G  measures how much the surface is curved in the x and y directions.
        H  measures the average of the two principal curvatures at each point on the surface.
        curvature coefficient K, measures the overall curvature of the surface at each point,
        with positive values indicating convex areas, negative values indicating concave areas, 
        and zero values indicating a flat surface.
        """
        dx, dy = np.gradient(surface)
        dxy, dxx = np.gradient(dx)
        dyy, _ = np.gradient(dy)
        G = (1 + dy ** 2) * dxx - 2 * dx * dy * dxy + (1 + dx ** 2) * dyy
        H = (dx ** 2 + dy ** 2 + 1) ** 1.5
        K = G / H
        return K

    def is_concave(self, K):
        """
        计算曲率半径，并判断曲面的凸凹性
        如果曲率半径中有任意一项小于零，则曲面被认为是凹的，返回True，否则返回False。
        当K> 0时，表示曲面局部上凸出，而K <0则意味着曲面凹陷。K = 0时，表明曲面在该点上是平的。
        K值的绝对值越大，表明该点上面的曲率越大，即弧线的半径越小，曲面越弯曲。因此，K值越大，曲面凸度越大。
        """
        # K = self.compute_gaussian_curvature(surface)
        R = 1 / K
        sub_n = R.shape[0] // 3
        R_sub = R[sub_n:-sub_n, sub_n:-sub_n]
        print(R_sub.mean(), R_sub.max(), R_sub.min())
        return (R < 0).any()
    
    def draw_3dcurve(self, x, y, z, xi, yi, zi, k):
        # 绘制拟合结果和高斯曲率
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(x, y, z, c='k')
        ax.plot_surface(xi, yi, zi, alpha=0.5, cmap='coolwarm')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax2 = fig.add_subplot(122)
        ax2.imshow(k, cmap='coolwarm')
        ax2.set_title('Gaussian Curvature')

        plt.savefig("gausian_curve_nose.png")
    
    def simulation_surface_delaunay(self, points):
        """
        points = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12],
                   [13, 14, 15]])
        """
        tri = Delaunay(points) # 三角剖分得到多面体
        # 计算每个三角面片的法向量
        normals = np.zeros(points.shape)
        for t in tri.simplices:
            a, b, c = points[t]
            normal = np.cross(b - a, c - a)
            normals[t] += normal
        # 通过PCA计算投影到二维平面上的主成分，使用法向量作为投影在平面上的压缩方向
        pca = PCA(n_components=2)
        pca.fit(normals)
        pc1, pc2 = pca.components_
        # 计算每个点在二维平面上的坐标，并将其作为曲面的参数
        params = np.dot(points, np.array([pc1, pc2]))

        # 创建一个仿射曲面模型，并使用参数拟合数据
        from sklearn.linear_model import LinearRegression

        model = LinearRegression()
        model.fit(params, points)
        # 用模型计算每个点的曲面法向量，并计算其主曲率
        for p in points:
            # 计算曲面法向量
            x, y = np.dot(p, np.array([pc1, pc2]))
            n = np.cross(model.coef_[0], model.coef_[1])
            normal = np.array([n[0]*pc1 + n[1]*pc2, n[2]])
            
            # 计算主曲率
            curvatures = np.linalg.eigvals(np.dot(tri.transform[t], np.vstack([normal, pc1]).T))
            kmin, kmax = min(curvatures), max(curvatures)

            # 计算曲面法向量和主曲率
    
    def resize_and_pad_image(self, image, size):
        """
        将图片按比例resize到1024*1024(size)，差的地方做pad处理
        """
        # 获取原图尺寸
        h, w = image.shape[:2]
        # 确定缩放比例
        scale = min(size[0]/w, size[1]/h)
        # 计算缩放后的尺寸
        new_size = (int(w*scale), int(h*scale))
        # 缩放图像
        resized_image = cv2.resize(image, new_size)
        # 计算需要填充的像素数
        pad_w = size[0] - new_size[0]
        pad_h = size[1] - new_size[1]
        # 计算填充的边界大小
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        # 填充图像
        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return padded_image
    
    def add_text_to_image(self, img, text_list, number_list, output_path):
        # 读取图片
        # img = cv2.imread(image_path)
        # 获取图片的高度和宽度
        import copy 
        img = copy.deepcopy(img)
        height, width = img.shape[:2]
        # if height <= 250 or width <= 250:
        #     # 图片小了，存在字体显示拥挤问题
        #     img = cv2.resize(img, (width*6, height*6))
        #     height, width = height*6, width*6
        #     # 可直接resize到1024, 400也存在显示问题
        # step_h = height // (28 * 2) if height >= 112 else 0.1
        if height >= 1024 or width >= 1024:
            pass
        else:
          img = self.resize_and_pad_image(img, (1024, 1024))
          height, width = img.shape[:2]
        step_h = height // (28 * 2)
        
        # 创建一个黑色的图像，用于放置文字
        text_img = np.zeros((height, width, 3), np.uint8)
        text_img[:] = (0, 0, 0)
        # 设置字体和字体大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        # 遍历文字列表和数字列表，按行写上文字和数字
        x = 10
        y = 10
        for i, text in enumerate(text_list):
            # 计算文字位置和数字位置
            y += step_h
            # 写上文字和数字
            cv2.putText(text_img, f"{text}:{number_list[i]:.3f}", (x, y), font, font_scale, (0, 0, 255), 1, cv2.LINE_AA)
        # 将文字图像和原图像叠加
        result = cv2.addWeighted(img, 0.3, text_img, 0.7, 0)
        # result = text_img
        if output_path.endswith("jpg"):
            output_path = output_path.replace("jpg", "png")
        # 保存结果到本地
        cv2.imwrite(output_path, result)
        
def get_mesh_world_location(imgf):
    import cv2
    import mediapipe as mp
    
    xx_locations = []  # world_loaction
    img_coordinates = []

    dframe = cv2.imread(imgf)
    image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                                refine_landmarks=True, min_detection_confidence=0.5)
    h, w, _ = dframe.shape
    results = face_mesh.process(image_input)
    base_name = os.path.basename(imgf).split(".")[0]
    for face in results.multi_face_landmarks:
        for landmark in face.landmark:
            x = landmark.x
            y = landmark.y
            z = landmark.z
            xx_locations.append([x, y, z])

            relative_x = int(x * w)
            relative_y = h - int(y * h)  # to x-y corrdinate
            img_coordinates.append([relative_x, relative_y, z])

            # cv2.circle(dframe, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)
        # cv2.imwrite(f'test_{base_name}.png', dframe)
    xx_locations = np.array(xx_locations)
    img_coordinates = np.array(img_coordinates)
    # print(img_coordinates.shape, xx_locations.shape)
    return xx_locations, img_coordinates

def draw_circle(idx_to_coordinates, img):
    for idx, landmark_px in idx_to_coordinates.items():
        img = cv2.circle(img, landmark_px, 1, (224, 224, 224), 1)
    return img

def draw_line(img, img_coordinates, lines_tuple):
    for ix, line in enumerate(lines_tuple):
        cv2.line(img, img_coordinates[line[0]],img_coordinates[line[1]], (0, 0, 255),1)
    return img

def draw_line_(img, line_tuple):
    for line in line_tuple:
        cv2.line(img, line, (0, 0, 255),1)

def iris_eraea(xx_locations, iris_indexs):
    pass

def test_x():
    import cv2
    import mediapipe as mp

    dframe = cv2.imread("./test_imgs/emo.jpg")

    image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                            min_detection_confidence=0.5)
    image_rows, image_cols, _ = dframe.shape
    results = face_mesh.process(cv2.cvtColor(image_input , cv2.COLOR_BGR2RGB))

    ls_single_face=results.multi_face_landmarks[0].landmark
    # for idx in ls_single_face:
        # print(idx.x,idx.y,idx.z)
    
    from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates

    for i, idx in enumerate(ls_single_face):
        cord = _normalized_to_pixel_coordinates(idx.x,idx.y,image_cols,image_rows)
        # print(cord)
        o = cv2.putText(dframe, f'{i}', cord, cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 0)
        # img = cv2.circle(o, cord, 1, (224, 224, 224), 1)
        cv2.imwrite('./test_emo_x.png', o)


if __name__ == '__main__':
    # imgf = "./test_imgs/brow_a1.jpg"
    # dframe = cv2.imread(imgf)
    # results = get_mesh_world_location(imgf)
    # # img = draw_circle(results[-1], dframe)
    # # cv2.imwrite(f'./circle_{i}.png', img)
    # # img = draw_line(dframe ,results[1], RIGHT_IRIS)
    # # img = draw_line(dframe ,results[1], LEFT_IRIS)
    # img = draw_line(dframe ,results[1], LEFT_EYEBROW+RIGHT_EYEBROW+LEFT_IRIS)
    # # img = draw_line(dframe ,results[1], LEFT_EYE + RIGHT_EYE)
    # # img = draw_line(dframe ,results[1], LIPS)
    # # img = draw_line(dframe ,results[1], OVAL)
    # # img = draw_line(dframe ,results[1], ALL_LINE)
    
    # cv2.imwrite('./line_iris_brow_a1_rirs.png', img)
    
    for i, imgf in enumerate(IMAGE_FILES):
        dframe = cv2.imread(imgf)
        dframe = resize_and_pad_image(dframe, (512, 512))
        h, w = dframe.shape[:2]
        # image_input = cv2.cvtColor(dframe, cv2.COLOR_BGR2RGB)
        results = get_mesh_world_location(imgf)
        print(results[0].shape, imgf, h, w)
        mbpp = MeshPoints2BoneParameters(results[1]) 
        r = mbpp.test_all_handle()  # TODO 结果需要在图像尺度上乘以一个系数，facemesh对图像处理，待考
        # print(r, imgf)
        imgname = os.path.basename(imgf)
        save_p = os.path.join("./test_result/parameter_add", "ptext_" + imgname)
        # print(len(r))
        # test transofmer for eye brow
        # A = mbpp.transformer_all_parameters(r)
        # r_brow = A @ np.array(r[:2])
        # print(r_brow)
        # r[:2] = r_brow
        r = mbpp.transformer_all_parameters(r)
        mbpp.add_text_to_image(dframe, mbpp.fp_name_en, r, save_p)
        # if i == 6:
        #     print(imgf, dframe.shape)
        #     r = mbpp.test_all_hand()
        #     print(r)
            # --------- test 2d curve simulation ---------
            # for j in range(4):
            #     x, y = results[-1][mbpp.eyebrow_index[j], 0],h -  results[-1][mbpp.eyebrow_index[j], 1]
            # # pp = [np.array([x[i], y[i]]) for i in range(len(x))]
            # # mbpp.bezier_curve_curvature(pp)
            #     mbpp.spline_curve(x, y, save_n="b"+str(j))
            # --------- test 3d curve simulation ---------
            # 选的点还需要精简一下
            # x = results[0][mbpp.nose_index[0], 0]
            # y = results[0][mbpp.nose_index[0], 1]
            # z = -results[0][mbpp.nose_index[0], 2]
            # print(x.shape, y.shape, z.shape)
            # mbpp.test_3d_simulation(x, y, z, n_grid=30)
            
            # --------- test mouth ---------
            
            # ------ test iris --------
            # area = mbpp.area_poly(results[0][mbpp.iris_index[1], :2])
            # print(results[0][mbpp.iris_index[1], :2].shape)
            # a = mbpp.traingle_area(results[0][mbpp.iris_index[1], :2][:3,:]) + mbpp.traingle_area(results[0][mbpp.iris_index[1], :2][[2,3,0],:])
            # # area = mbpp.area_poly(results[1][mbpp.iris_index[0], :2])
            # print(area, '面积', a)
            
            # -------- test eye ---------
            
            # ps = [(x[i], y[i]) for i in range(4)]
            # ps_pair = [(ps[i-1], ps[i]) for i in range(4)]
            # g = draw_line_(dframe, ps_pair)
            # cv2.imwrite("xx_g.png", g)
    print()
    # test_x()
    