import json
import numpy as np
import csv
import cv2
import os,ast
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


###load marker json file

jsonFromFourDiseaseModel ='via_project_5Sep2019_12h9m.json'
with open(jsonFromFourDiseaseModel, "r") as read_file:
    data1 = json.load(read_file)




### converting polygon points to rectangle bounding boxes

class BoundingBox(object):
    """
    A 2D bounding box
    """
    def __init__(self, points):
        if len(points) == 0:
            raise ValueError("Can't compute bounding box of empty list")
        self.minx, self.miny = float("inf"), float("inf")
        self.maxx, self.maxy = float("-inf"), float("-inf")
        for x, y in points:
            # Set min coords
            if x < self.minx:
                self.minx = x
            if y < self.miny:
                self.miny = y
            # Set max coords
            if x > self.maxx:
                self.maxx = x
            elif y > self.maxy:
                self.maxy = y
    @property
    def width(self):
        return self.maxx - self.minx
    @property
    def height(self):
        return self.maxy - self.miny
    def __repr__(self):
        return "BoundingBox({}, {}, {}, {})".format(
            self.minx, self.maxx, self.miny, self.maxy)


img_list = []

#######################################################
for (key1,val1) in data1.items():

    _via_img_metadata = data1['_via_img_metadata']

    for (key,val) in _via_img_metadata.items():

        dict_img_data ={}

        info = _via_img_metadata[key]

        filename = info['filename']

        dict_img_data['filename'] = filename

        # count = count+1
        # print(count)
        # print(info['filename'])

        region_prop = info['regions']

        ### collect all x_points and y_points of polygon
        x_coordiate = np.asarray(region_prop[0]['shape_attributes']['all_points_x'])
        y_coordiate = np.asarray(region_prop[0]['shape_attributes']['all_points_y'])

        ##read image using opencv
        img  = cv2.imread('/media/guruleben/datasetsRiyaj/images/'+filename)
        img_name = os.path.splitext(os.path.basename(filename))

        
        ##define size of new images 
        new_height,new_width = (608,608)

        ## find scaling factor for image height and width so we can rescale corrosponding 
        ## polygon/bounding box coordinate
        h_factor = new_height/img.shape[0]
        w_factor = new_width/img.shape[1]

        img = cv2.resize(img,(608,608))
        mask = np.zeros((img.shape))
        # print(x_coordiate*w_factor,y_coordiate*h_factor)

        x_coordiate,y_coordiate = x_coordiate*w_factor,y_coordiate*h_factor

        ## vertically stack x and y points to draw polygon around object to be detected

        pts = np.vstack((x_coordiate,y_coordiate)).astype(np.int32).T
        
        ## find bounding box coordinates
        x1 = BoundingBox(pts).minx
        # x1  = int(x1 * w_factor)
        x1  = int(x1 )
        y1 = BoundingBox(pts).miny
        # y1  = int(y1 * h_factor)
        y1  = int(y1)
        x2 = BoundingBox(pts).maxx
        # x2  = int(x2 * w_factor)
        x2  = int(x2 )
        y2 = BoundingBox(pts).maxy
        # y2  = int(y2 * h_factor)
        y2  = int(y2)  
        # print(filename ,x1,y1,x2,y2)

        ### draw polygon using opencv
        cv2.polylines(img,  [pts],  False,  (0,255,0),  2)

        ### To create mask for semantic seg if needed. and draw contours using opencv
        ## fillregion inside
        # mask = cv2.drawContours(mask, [pts], -1, (255,255,255), -1)
        cv2.fillPoly(mask, pts =[pts], color=(255,255,255))
        img = cv2.drawContours(img, [pts], -1, (255,0,0), 3)
        
        ## Approximate rectangle using polygon points
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow('img',img)
        cv2.waitKey(0)
        # cv2.imwrite(img_name[0]+'_mask'+img_name[1],mask)
        # cv2.imwrite(img_name[0]+'_resized'+img_name[1],img)
        img_list.append(img)
        # dict_img_data['height'] = 608
        # dict_img_data['width'] = 608
        # dict_img_data['ann'] = {'bboxes':np.array([x1,y1,x2,y2],dtype = np.float32),'labels': np.array([1],dtype=np.int64)}
        # data_list.append(dict_img_data)
        # print(data_list)


#### Displaying images
c = 1
plt.figure(figsize=[20, 20])
for img_ in img_list[:10]:
    # print(img)
    plt.subplot(2, 5, c)
    plt.imshow(img_)
    plt.title("Image %s" % c)
    c += 1
    
plt.show()


