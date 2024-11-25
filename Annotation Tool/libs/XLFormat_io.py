#!/usr/bin/env python
# -*- coding: utf8 -*-
import codecs
import os

from libs.constants import DEFAULT_ENCODING
import XenoWareFormat as xw

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING
root_dir = os.getcwd()
class_file = os.path.join(root_dir, './data/predefined_classes.txt')

# This is just an example to test
# base_dir = '.\\ToYash'
# txt_dir = os.path.join(root_dir, base_dir)
# files = glob.glob(os.path.join(txt_dir, '*.txt'))
# file1490 = files[0]

'''
    What is XLFormat?
        - xmin,ymin,xmax,ymax,label
    
    This scripts helps us to load the label file with XL Format and also write to the same format
    
    This script contains following classes:
        1. XLReader
        2. XLWriter
        
    1. XLReader:
        Reads and parses the text file. 
        Gets the xmin, ymin, xmax, ymax, class.
        
    2. XLWriter:
        Creates the label file with the same name as the binary file.
        Save the coordinates of the bounding boxes with XLFormat in the label file.
'''


class XLReader:

    def __init__(self, filepath, image, classListPath=None):
        # shapes type: [label, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]]
        self.shapes = []
        self.filepath = filepath

        # Get class list path
        if classListPath is None:
            self.classListPath = class_file
        else:
            self.classListPath = classListPath

        # Open the classes file
        classesFile = open(self.classListPath, 'r')
        self.classes = classesFile.read().strip('\n').split('\n')

        self.verified = False
        try:
            self.parseXLFormat()
        except:
            pass

    def getShapes(self):
        return self.shapes

    def addShape(self, label, xmin, ymin, xmax, ymax, difficult):
        # Adding shapes in this format
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        self.shapes.append((label, points, None, None, difficult))
        
    def read_matrix(self, file_name):
        # data = np.fromfile(file_name, dtype='<d')  # little-endian double precision float
        # nr_rows = 512
        # nr_cols = int(len(data) / nr_rows)
        # img = data.reshape((nr_rows, nr_cols))
        c, m = xw.XW_ReadFile(file_name)
        img = c['data']
        return img        
    
    
    def read_pointCloud_distance(self,filepath):
        pnt_ = self.read_matrix(filepath)
        points=[]
        #5137*13
        for i in range(pnt_.shape[0]):
             
            x = pnt_[:,4:6][i][0]
            y = pnt_[:,4:6][i][1]
            d = pnt_[i,6]

            if x<0 or y<0:
                pass
            else:
                points.append((x,y,d))
                
        return points

    def parseXLFormat(self):
        # Open bounding box file and parse it to add shapes in the aforementioned function
        
        bndBoxFile = open(self.filepath, 'r')
        for idx in bndBoxFile:
            bndBox = idx.strip().split(' ')[0].split(',')
            xmin, ymin, xmax, ymax, classIndex = [int(z) for z in bndBox]
            label = self.classes[int(classIndex)]
            self.addShape(label, xmin, ymin, xmax, ymax, False)
            
            
        


class XLWriter:

    def __init__(self, foldername, filename, databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult):
        # Adds the bounding boxes to the list
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        self.boxlist.append(bndbox)

    def BndBoxToXLFormat(self, box, classList=[]):
        # Convert the Bounding box to XL Format and return it
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

        # PR387
        boxName = box['name']
        if boxName not in classList:
            classList.append(boxName)

        classIndex = int(classList.index(boxName))

        return classIndex, xmin, ymin, xmax, ymax

    def save(self, classList=[], targetFile=None):
        # Save the label file with the correct bounding box coordinates
        if targetFile is None:
            out_file = open(
                self.filename + TXT_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        for box in self.boxlist:
            classIndex, xmin, ymin, xmax, ymax = self.BndBoxToXLFormat(box, classList)
            out_file.write("%d,%d,%d,%d,%d\n" % (xmin, ymin, xmax, ymax, classIndex))

        out_file.close()
