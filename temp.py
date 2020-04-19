import os
import sys
import argparse
import numpy as np
import cv2
import random
import math
import dlib
from lxml import etree
import xml.etree.cElementTree as ET
from PIL import Image, ImageFile
import pandas as pd
__version__ = '0.3.0'


IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
# IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')


center_1=[]
center_2=[]
mask_left=[]
mask_right=[]
new_heigh=[]
chin_right=[]
x_ori=[]
y_ori=[]
rect_top = []
rect_bottom = []
w1 = []
h1 = []
length=0
output = []
mask_num=[]

def write_xml(folder, image_name, image_shape, objects, tl, br, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

#    image = cv2.imread(img.path)
    height, width, depth = image_shape

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    for obj, topl, botr in zip(objects, tl, br):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(topl[0])
        ET.SubElement(bbox, 'ymin').text = str(topl[1])
        ET.SubElement(bbox, 'xmax').text = str(botr[0])
        ET.SubElement(bbox, 'ymax').text = str(botr[1])

    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)
    save_path = os.path.join(savedir, image_name.replace('jpg', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    print('face_co-ordinates '+str(((rect[3],rect[0]),(rect[1],rect[2]))))
    x_ori.append(rect[3])
    y_ori.append(rect[0])
    w1.append(w)
    h1.append(h)
    rect_top.append((rect[3],rect[0]))
    rect_bottom.append((rect[1],rect[2]))
    return (x, y, w, h)


def face_alignment(faces):
    # Forecast key points
    predictor = dlib.shape_predictor('dlib/shape_predictor_68_face_landmarks.dat')
    faces_aligned = []
    for face in faces:
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y

        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy, dx) * 180. / math.pi

        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)

        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned


def cli(pic_path = '1.thr_supporting_actors_group_9901_17_0984.jpg',save_pic_path = ''):
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    # parser.add_argument('pic_path', default='/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',help='Picture path.')
    # parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print('fPicture {pic_path} not exists.')
        sys.exit(1)

    # if args.black:
    #     mask_path = BLACK_IMAGE_PATH
    # elif args.blue:
    #     mask_path = BLUE_IMAGE_PATH
    # elif args.red:
    #     mask_path = RED_IMAGE_PATH
    # else:
    #     mask_path = DEFAULT_IMAGE_PATH
    mask_path = []
    for dirs,subdirs,files in os.walk(IMAGE_DIR):
        for name in files:
            mask_path.append(IMAGE_DIR+'/'+str(name))
    length = len(mask_path)



    FaceMasker(pic_path, mask_path, True, 'hog',save_pic_path).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog',save_path = ''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img = None
        self._mask_img = None

    def mask(self):
        import face_recognition

        face_image_np = face_recognition.load_image_file(self.face_path)
        img = cv2.imread(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            #print('face_image_np ' + str(face_image_np))
            #print('face_locations ' + str(face_locations))
            self._mask_face(face_landmark)


        if found_face:
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            for (i, rect) in enumerate(face_locations):
                src_face_num = src_face_num + 1
                ## face coordinate for every face in image
                (x, y, w, h) = rect_to_bbox(rect)
                #print('(x,y,w,h) '+str((x,y,w,h)))
                detect_face = with_mask_face[y:y+h,x:x+w]
                src_faces.append(detect_face)
            # Face alignment operation and save
            faces_aligned = face_alignment(src_faces)
            file_name = self.face_path.split('/')[-1].split('.jpg')[0]
            #print('file_name '+file_name)
            face_num=0
            for faces in faces_aligned:
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                size = (int(128), int(128))
                #faces_after_resize = cv2.resize(faces, size, interpolation=cv2.INTER_AREA)


                left_x = center_1[face_num]-mask_left[face_num]
                left_y = center_2[face_num]-new_heigh[face_num]//2
                right_x = center_1[face_num]+mask_right[face_num]
                right_y = center_2[face_num]+new_heigh[face_num]//2

                #Normalisation
                left_x = max(0,left_x-x_ori[face_num])
                left_y -= y_ori[face_num]
                right_x = min(right_x-x_ori[face_num],chin_right[face_num]-x_ori[face_num],w1[face_num]-1)
                right_y = min(h1[face_num]-1,right_y-y_ori[face_num])
                # mask coordinate
                print("Mask coordinates " + str((left_x, left_y)) + str((right_x, right_y)))

                #cv2.rectangle(img,rect_top[face_num], rect_bottom[face_num], (0, 255, 0), 2)
                #cv2.rectangle(faces, (left_x, left_y), (right_x, right_y), (0, 255, 0), 2)
                eee = cv2.imwrite(self.save_path+str(file_name)+'_'+str(face_num)+'.jpg',faces)
                write_xml("trainingmask", self.save_path+str(file_name)+'_'+str(face_num)+'.jpg',[abs(rect_top[face_num][0]-rect_bottom[face_num][0]),abs(rect_top[face_num][1]-rect_bottom[face_num][1]),3],['mask'], [(left_x, left_y)],[(right_x, right_y)],"annotation1/")
                li = []
                li.append(left_x)	#mask coordinate
                li.append(left_y)
                li.append(right_x)
                li.append(right_y)
                li.append(rect_top[face_num][0])	#face coordinate whole image
                li.append(rect_top[face_num][1])	#face coordinate whole image
                li.append(rect_bottom[face_num][0])
                li.append(rect_bottom[face_num][1])	
                li.append(mask_num[face_num])		#to check randomness 
                output.append(li)
                face_num = face_num + 1
            cv2.imwrite(self.save_path+str(file_name)+'_'+str(face_num+1)+'.jpg',img)
            unmasked_face = ['unmasked']*face_num
            topl=[]
            botr=[]
            for i in range(face_num):
                topl.append((rect_top[i][0],rect_top[i][1]))
                botr.append((rect_bottom[i][0], rect_bottom[i][1]))
            write_xml("trainingmask",self.save_path+str(file_name)+'_'+str(face_num+1)+'.jpg',img.shape, unmasked_face, topl, botr, "annotation2/")
            center_1.clear()
            center_2.clear()
            mask_left.clear()
            mask_right.clear()
            new_heigh.clear()
            chin_right.clear()
            x_ori.clear()
            y_ori.clear()
            rect_top.clear()
            rect_bottom.clear()
            w1.clear()
            h1.clear()
            # if self.show:
            #     self._face_img.show()
            # save
            # self._save()
        else:
            #Record uncut pictures here
            print('Found no face.'+self.save_path)

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        #print('chin_left '+str(chin_left_point))
        chin_right_point = chin[chin_len * 7 // 8]
        #print('chin_right '+str(chin_right_point))
        # split mask and resize
        a = []
        for i in range(9):
            a.append(random.choice(self.mask_path))
        random.shuffle(a)
        num = a[0]
        mask_num.append(num)
        self._mask_img = Image.open(num)
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))
        #print('new_height '+str(new_height))
        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        #print('(mask_left_img.width,mask_right_img.width) '+str((mask_left_img.width,mask_right_img.width)))
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        #print('(center_x,center_y) '+str((center_x,center_y)))
        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)
        #print('(box_x, box_y) '+str((box_x, box_y))+str((mask_img.width,mask_img.height)))
        #print('mask_img '+str((mask_img.width,mask_img.height)))

        mask_left.append(mask_left_img.width)
        mask_right.append(mask_right_img.width)
        chin_right.append(chin_right_point[0])
        center_1.append(center_x)
        center_2.append(center_y)
        new_heigh.append(new_height)
    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print('Save to '+new_face_path)

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    dataset_path = 'trainingimages/'
    save_dataset_path = 'trainingmask/'
    global face_num
    face_num=0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            new_root = root.replace(dataset_path, save_dataset_path)
            # if not os.path.exists(new_root):
            #     os.makedirs(new_root)
            # deal
            imgpath = os.path.join(root, name)
            save_imgpath = os.path.join(new_root,'')
            print('imgpath '+str(imgpath))
            cli(imgpath,save_imgpath)
    la = pd.DataFrame(output)
    la.to_csv(save_dataset_path+'tmp.csv',index=False,header=False)
