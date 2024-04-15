import math
import csv
import cv2, os
import json



class Cell_Mask():
    '''
       This is the cell  mask class. It will contain a label, a bounding box,
       and all the xy positions of the mask
       '''

    def __init__(self, label = "Cell"):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        self.verticies = []
        self.xposes = []
        self.yposes = []
        self.xmax = -1
        self.ymax = -1
        self.xmin = -1
        self.ymin = -1
        self.bounding_box = set()
        self.label = label
        self.sum = -1
        self.cell_num = -1
        self.area = 0
        self.marked = False
        self.center = 0
        self.name = "Cell"

    def calculate_bb(self):
        # make sure this does not ex
        self.xmax = max(self.xposes)
        self.ymax = max(self.yposes)
        self.xmin = min(self.xposes)
        self.ymin = min(self.yposes)
        self.bounding_box.add(min(self.xposes))
        self.bounding_box.add(min(self.yposes))
        self.bounding_box.add(max(self.xposes))
        self.bounding_box.add(max(self.yposes))
        self.area = (self.ymax - self.ymin) * (self.xmax - self.xmin)


    def inside_box(self, xmax, ymax, xmin, ymin):
        '''
        This method make sure if two cells are within one another they are the same cell and are added as xypositions
        in the list.
        :param bounding_box2:
        :return:
        '''
        if self.xmin > xmin and self.ymin > ymin and self.xmax < xmax and self.ymax < ymax :
            return True
        else:
            return False


    def print_mask(self):
        '''
        This method prints the masks.
        :return:
        '''
        print("MASK LENGTH: " + str(len(self.xposes)))
        print("X Y POSITIONS:")
        xpos = list(self.xposes)
        ypos = list(self.yposes)
        #print(xpos)
        #print(ypos)
        print("Label:" + self.label)
        print("Bounding Box:" + str(self.bounding_box))
        print("YMIN: " + str(self.xmin))
        print("YMAX: " + str(self.xmax))
        print("XMIN: " + str(self.ymin))
        print("XMAX: " + str(self.ymax))
        print("AREA: " + str(self.area))

    def sections(self,list_of_pos):
        # this will be a dictionary of lists
        sections = dict()
        section = 0

        prev_vertex = list_of_pos[0]
        sections[0] = []
        sections[0].append(list_of_pos[0])
        # loop through each finding the sections
        for index in range(0, len(list_of_pos)):
            if (list_of_pos[index].dist_from_center([prev_vertex.xpos, prev_vertex.ypos]) > 2.0):
                section = section + 1
                sections[section] = []
                sections[section].append(list_of_pos[index])
            sections[section].append(list_of_pos[index])
            prev_vertex = list_of_pos[index]
        print("SECTION:" + str(section))

        all_sections = []
        for i in range(0, section + 1):
            all_sections.append(i)
        # loop through each section until we find each
        sorted = []

        # start at 0
        sorted = sorted + sections[0]
        all_sections.remove(0)
        reversed = False
        while len(all_sections) > 0:
            first_section = all_sections[0]
            if(len(sections[first_section]) < 3):
                all_sections.remove(first_section)
                continue

            shortest_dist = sorted[-1].dist_from_center([sections[first_section][0].xpos, sections[first_section][0].ypos])
            shortest_sec = first_section
            for sec in all_sections:
                distance = sorted[-1].dist_from_center([sections[sec][0].xpos, sections[sec][0].ypos])
                distance_rev = sorted[-1].dist_from_center([sections[sec][-1].xpos, sections[sec][-1].ypos])
                reversed = False
                if distance <= shortest_dist:
                    shortest_dist = distance
                    shortest_sec = sec
                    reversed = False
                if distance_rev < shortest_dist:
                    shortest_dist = distance_rev
                    shortest_sec = sec
                    reversed = True

            if reversed:
                sections[shortest_sec].reverse()

            sorted = sorted + sections[shortest_sec]
            all_sections.remove(shortest_sec)
        return sorted

    def create_json_dict(self):
        '''
        This method creates the json dictionary
        :return:
        '''
        json_dict = dict()
        list_of_pos = []
        prev_point = Vertex(self.yposes[0], self.xposes[0])
        jump_point = len(self.xposes)
        jump_found = False
        for index in range(0, len(self.xposes)):
            # find the first jump and then use that as a stopping point.
            point = Vertex(self.yposes[index], self.xposes[index])
            # print(prev_point.distance(point))
            if prev_point.distance(point) > 20.0:
                if not jump_found:
                    jump_point = index
                    jump_found = True
            prev_point = point
            list_of_pos.append(point)
        x_y = []
        for index in range(0,jump_point):
            vertex = list_of_pos[index]
            x_y.append((vertex.xpos, vertex.ypos))

        json_dict["points"] = x_y
        json_dict["label"] = self.label
        json_dict["cell number"] = self.cell_num

        # look at best way to make this json
        return json_dict

class Vertex():
    '''
    This is the vertex class, it contains its edges as well as its x and y postion
    '''
    def __init__(self, xpos, ypos):
        '''
        Set edges to default to nothing.
        :param xpos: pixel location
        :param ypos: pixel location
        :param edges: what vertecies it connects to
        '''
        self.xpos = xpos
        self.ypos = ypos
        self.edges = set()
        self.marked = False
        self.visited = False
        self.center_dist = self.dist_from_center([0,0])

    def add_edge(self, vertex):
        '''
        Add a vertex to the edge set
        :param vertex:
        :return:
        '''
        self.edges.add(vertex)

    def distance(self, vertex):
        '''
        This method will find the distance between this vertex and another vertex
        :param center:
        :return:
        '''
        return math.sqrt((self.xpos - vertex.xpos) ** 2 + (self.ypos - vertex.ypos) ** 2)

    def dist_from_center(self, center):
        '''
        This method will find the distance between this vertex and another vertex
        :param center:
        :return:
        '''
        return math.sqrt((self.xpos - center[0]) ** 2 + (self.ypos - center[1]) ** 2)





def is_interest_point(image, i, j):
    '''
    This method checks to see if a pixel is part of the edge of a mask
    :param image:
    :param i:
    :param j:
    :return:
    '''
    if image[i][j] == 0:
        if(image[i][j - 1] == 255 or image[i][j + 1] == 255 or image[i + 1][j] == 255 or image[i - 1][j] == 255):
            return True
    else:
        return False

def find_edges(image, i, j , point_dict):
    '''
    This method finds creates an edge for each connected edge pixel
    :param image:
    :param vertex:
    :return:
    '''
    vertex = Vertex(i, j)
    x = vertex.xpos
    y = vertex.ypos

    safe_x = []
    safe_y = []
    changes = [-1, 1]
    for change in changes:
        if is_interest_point(image, x + change, y):
            i = x + change
            j = y
            key = str(i) + ":" + str(j)
            iv = Vertex(i, j)
            vertex.add_edge(iv)
            safe_x.append(i)
            safe_y.append(j)
        if is_interest_point(image, x, y + change):
            i = x
            j = y + change
            key = str(i) + ":" + str(j)
            iv = Vertex(i, j)
            vertex.add_edge(iv)
            safe_x.append(i)
            safe_y.append(j)

    changes = [-1, 1]
    for change in changes:
        if is_interest_point(image, x + change, y + changes[0]):
            i = x + change
            j = y + changes[0]
            iv = Vertex(i, j)
            vertex.add_edge(iv)
        if is_interest_point(image, x + change, y + changes[1]):
            i = x + change
            j = y + changes[1]
            key = str(i) + ":" + str(j)
            iv = Vertex(i, j)
            vertex.add_edge(iv)


    return vertex


def make_graph(image,h,w, point_dict):
    '''
    Input is a binary image. An Interest point is a point where it goes white to black or black to white.
    this function will loop through all of the pixels and set up the graph.
    :param image:
    :param h:
    :param w:
    :return:
    '''
    # loop through each pixel in the image
    # we may need to add a white border to the image for this to work perfectly
    for i in range(1,h-1):
        for j in range(1,w-1):
            # if it is already in the dict of points, check to see if it has been marked, and
            # then if it has not been marked, find all the edges for that point.
            key = str(i) + ":" + str(j)
            if key in point_dict:
                if point_dict[key].marked:
                    continue
                else:
                    iv = find_edges(image, i,j, point_dict)
                    iv.marked = True
                    point_dict[key] = iv
            elif is_interest_point(image,i,j):
                iv = find_edges(image, i,j, point_dict)
                # after we find the edges mark that we have found the edges
                iv.marked = True
                key = str(i) + ":" + str(j)
                point_dict[key] = iv

    return point_dict

def dfs(start_vertex, point_dict):
    '''
    This will run a dfs to find a cycle, which is equal to a
    :param start_vertex:
    :return:
    '''
    # try to indroduce relation trees here. perhaps trees do the trick.
    # add the spot to the bag
    cur_mask = Cell_Mask()
    bag = []
    start_key = str(start_vertex.xpos) + ":" + str(start_vertex.ypos)
    bag.append(start_key)
    first = True
    end_key = ""
    while (len(bag) > 0):
        # remove from bag
        vertex_key = bag.pop()
        if (not point_dict[vertex_key].visited):
            # mark the vertex
            point_dict[vertex_key].visited = True
            # add the point to the mask_points
            cur_mask.xposes.append(point_dict[vertex_key].xpos)
            cur_mask.yposes.append(point_dict[vertex_key].ypos)
            # loop through each of the edges
            for edge in point_dict[vertex_key].edges:
                edge_key = str(edge.xpos) + ":" + str(edge.ypos)
                # hopefully this will stop early.
                if edge_key == end_key:
                    cur_mask.xposes.append(point_dict[end_key].xpos)
                    cur_mask.yposes.append(point_dict[end_key].ypos)
                    return cur_mask, point_dict, True
                if first:
                    end_key = edge_key
                    first = False
                if point_dict[edge_key].visited:
                    continue
                bag.append(edge_key)

    return cur_mask, point_dict, False

def find_masks(labels, point_dict):
    '''
    This method drives the dfs, and loops through each
    :param labels:
    :return:
    '''
    cur_masks = dict()
    key_val = 0
    for key in point_dict.keys():
        vertex = point_dict[key]
        if not vertex.visited:
            mask, point_dict_new, should_add = dfs(vertex, point_dict)
            # if should_add == True:
            point_dict = point_dict_new
            cur_masks[key_val] = mask
            key_val = key_val + 1


    masks = []

    for key in cur_masks.keys():
        cur_masks[key].calculate_bb()
        cur_masks[key].sum = sum(cur_masks[key].bounding_box)
        if cur_masks[key].area > 4:
            masks.append(cur_masks[key])

    masks.sort(key= lambda x:x.area, reverse=False)

    final_masks = []
    cell_num = 0
    # new algo for seeing if boxes are contained
    while(len(masks) > 0):
        top_mask = masks.pop()
        if top_mask.marked:
            continue
        # check each remaining mask combo to see if it is in the box
        for mask in masks:
            if mask.inside_box(top_mask.xmax, top_mask.ymax, top_mask.xmin, top_mask.ymin):
                # this will add the holes. other wise it is ignored.
                ####### CHANGE THIS BACK #######
                # top_mask.xposes = top_mask.xposes + mask.xposes
                # top_mask.yposes = top_mask.yposes + mask.yposes
                mask.label = "Hole"
                # final_masks.append(mask)
                mask.marked = True
        cell_num += 1
        top_mask.cell_num = cell_num
        top_mask.label = "Cell"
        final_masks.append(top_mask)

    final_masks.sort(key=lambda x: list(x.bounding_box)[0], reverse=False)
    print("ORIG")
    index = 0
    for mask in final_masks:
        if not index >= len(labels):
            mask.label = labels[index]
            index = index + 1
        mask.print_mask()

    return final_masks



def label_image(image, labels):
    '''
    This is the driver for creating json image labels. It generates the grpah, and then
    runs dfs on that graph to properly allocate
    :param image:
    :param labels:
    :return:
    '''
    point_dict = dict()
    h = image.shape[0]
    w = image.shape[1]
    point_dict = make_graph(image, h, w, point_dict)
    cell_list = find_masks(labels, point_dict)
    cell_dict = dict()
    shapes = []
    for mask in cell_list:
        mask_dict = mask.create_json_dict()
        # makes sure it is a polygon
        if len(mask_dict["points"]) < 5:
            continue
        shapes.append(mask_dict)
    cell_dict["shapes"] = shapes
    return cell_dict
    # add to json


def tiff_to_png(image_path):
    # convert image and return
    image = cv2.imread(image_path, 0)
    cv2.imshow("input image", image)
    # define a threshold, 128 is the middle of black and white in grey scale
    thresh = 128
    # threshold the image
    img_binary = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return img_binary


def make_json(path, json_pth, labels):
    '''
    This method creates json for 1 image. It is used in main to create for batches as well.
    :param path:
    :return:
    '''
    # print("Input image path:")
    # path = input()
    # change to black and white
    image = tiff_to_png(path)
    # print("NUMBER OF CELLS")
    # cell_count = int(input())

    # print("left to right top to bottom cell labels:")
    # for cell in range(0,cell_count):
    #     print("cell #" + str(cell) + ":")
    #     labels.append(input())

    cell_dict = label_image(image, labels)
    cell_dict["imagePath"] = path.replace('_mask', '').replace('.tif', '.png')
    cell_dict["imageHeight"] = image.shape[0]
    cell_dict["imageWidth"] = image.shape[1]
    # write the json
    with open(json_pth, "w") as write:
        json.dump(cell_dict, write)

def main():
    '''
    The main method takes user input for image path, and saves the json image in the same folder.
    :return:
    '''

    # parse csv and create csv dictionary.
    with open("/Users/tom/Desktop/Stanford/RA/OligodendroSight/mrcnn/bounding_box_labels.csv") as file:
        heading = next(file)

        reader = csv.reader(file)
        label_dict = dict()
        for row in reader:
            label_dict[row[0] + "_mask.tif"] = row[1:]

    print(label_dict)
    for key in label_dict.keys():
        label_dict[key] = ' '.join(label_dict[key]).split()

    print(label_dict)

    # get the path to the data
    # print("Batch or One Image?")
    # num_vals = input()

    ### CHANGE DIR PATH HERE ###
    dir_path = "/Users/tom/Desktop/Stanford/RA/OligodendroSight/mrcnn/data/masks"

    # loop through each image in a directory.
    image_list = []
    index = 1
    for filename in [file for file in os.listdir(dir_path)]:
        image_pth = os.path.join(dir_path, filename)
        if image_pth.__contains__('281.it11.20x.r1.4.tif'):
            print("IMAGE #: " + str(index))
            print(filename)
            if filename in label_dict.keys():
                labels = label_dict[filename]
            else:
                labels = []
            image_name = filename.replace('.tif','.json')
            json_pth = os.path.join(dir_path, image_name)
            make_json(image_pth,json_pth, labels)
            index = index + 1



if __name__ == '__main__':
    main()
