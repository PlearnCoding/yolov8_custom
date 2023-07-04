# -*- coding: utf-8 -*-

from xml.dom import minidom
import os
import glob



def convert_coordinates(size, box):
    dw = 1.0/size[0]
    dh = 1.0/size[1]
    x = (box[0]+box[1])/2.0
    y = (box[2]+box[3])/2.0
    w = box[1]-box[0]
    h = box[3]-box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)


def convert_xml2yolo( lut,xml_path,img_path,lbl_path,outfile ):
    fname_list =[]
    for fname in glob.glob(xml_path+"*.xml"):
        
        xmldoc = minidom.parse(fname)
        fname_s = fname.split(os.path.sep)[-1]
        fname_t = (fname_s[:-4]+'.txt')
        fname_out = lbl_path+fname_t
        fname_img = fname_s[:-4]+'.jpg'
        fname_img = img_path+fname_img
        print('read file : ',fname)
        with open(fname_out, "w") as f:

            itemlist = xmldoc.getElementsByTagName('object')
            size = xmldoc.getElementsByTagName('size')[0]
            width = int((size.getElementsByTagName('width')[0]).firstChild.data)
            height = int((size.getElementsByTagName('height')[0]).firstChild.data)

            for item in itemlist:
                # get class label
                classid =  (item.getElementsByTagName('name')[0]).firstChild.data
                if classid in lut:
                    label_str = str(lut[classid])
                else:
                    label_str = "-1"
                    print ("warning: label '%s' not in look-up table" % classid)

                # get bbox coordinates
                xmin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmin')[0]).firstChild.data
                ymin = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymin')[0]).firstChild.data
                xmax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('xmax')[0]).firstChild.data
                ymax = ((item.getElementsByTagName('bndbox')[0]).getElementsByTagName('ymax')[0]).firstChild.data
                b = (float(xmin), float(xmax), float(ymin), float(ymax))
                bb = convert_coordinates((width,height), b)
                #print(bb)

                f.write(label_str + " " + " ".join([("%.6f" % a) for a in bb]) + '\n')
        fname_list.append(fname_img)
        print ("wrote %s" % fname_out)
        with open(outfile, "w") as f:
            f.write('\n'.join(f for f in fname_list))
        print ("wrote %s" % outfile)



def main():
    #classes={'Dent':0,'PinHole':1,'Gas':2,'Slag':3,'Shrinkage':4,'SandDrop':5,'SandBroken':6,'Other':7}
    #classes={'Dent':0,'PinHole':0,'Gas':0,'Slag':0,'Shrinkage':0,'SandDrop':0,'SandBroken':0,'Other':0}
    classes={'abs-lh':0,'abs-rh':1,'non-lh':0,'non-rh':1}
    #classes = {'cylinder':0}
    xml_path = './resizedrh/' # position stored xml
    img_path = './resizedrh/' # position stored img
    lbl_path = './resizedrh/'
    outfile = './resizedrh/train_rh.txt'
    convert_xml2yolo( classes,xml_path,img_path,lbl_path,outfile )
    #xml_path = './datasets-knuckle/annotations/val/'
    #img_path = './datasets-knuckle/images/val/'
    #lbl_path = './datasets-knuckle/labels/val/'
    #outfile = './datasets-knuckle/labels/val_knuckle.txt'
    #convert_xml2yolo( classes,xml_path,img_path,lbl_path,outfile )


if __name__ == '__main__':
    main()
