#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import WheelEncoderStamped
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage, Image
from duckietown.utils.image.ros import compressed_imgmsg_to_rgb, rgb_to_compressed_imgmsg

import math

import tensor_helper

class TensorNode(DTROS):
    """
        
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(TensorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.PERCEPTION
        )
        # get the name of the robot
        self.veh = rospy.get_namespace().strip("/")

        # Construct publishers
        self.inference_pub = rospy.Publisher(
            f'/{self.veh}/tensor_node/compressed',
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION
        )

        # Camera subscriber:
        camera_topic = f'/{self.veh}/camera_node/image/compressed'
        _ = rospy.Subscriber(
            camera_topic,
            CompressedImage, 
            self.cbImage, 
            buff_size=10000000, 
            queue_size=1
        )

        self.engine = tensor_helper.get_engine("yolov4_1_3_416_416_fp16.engine")
        self.contect = self.engine.create_execution_context()
        self.buffers = tensor_helper.allocate_buffers(self.engine, 1)
        IN_IMAGE_H, IN_IMAGE_W = 416, 416
        self.context.set_binding_shape(0, (1, 3, IN_IMAGE_H, IN_IMAGE_W))
        self.num_classes = 80
        namesfile = 'coco.names'
        self.class_names = tensor_helper.load_class_names(namesfile)


        self.log("Initialized!")

    def cbImage(self, msg):
        img = compressed_imgmsg_to_rgb(msg)
        self.inferPublisher(img)


    def inferPublisher(self, img):
        boxes = tensor_helper.detect(self.context, self.buffers, img, (416,416), 80)
        infer = self.plot_boxes_cv2(img, boxes[0], class_names=class_names)
        
        cmprsmsg = rgb_to_compressed_imgmsg(infer)
        self.inference_pub.publish(cmprsmsg)
    
    def plot_boxes_cv2(self, img, boxes, class_names=None, color=None):
        import cv2
        colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], dtype=np.float32)

        def get_color(c, x, max_val):
            ratio = float(x) / max_val * 5
            i = int(math.floor(ratio))
            j = int(math.ceil(ratio))
            ratio = ratio - i
            r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]
            return int(r * 255)

        width = img.shape[1]
        height = img.shape[0]
        for i in range(len(boxes)):
            box = boxes[i]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)

            if color:
                rgb = color
            else:
                rgb = (255, 0, 0)
            if len(box) >= 7 and class_names:
                cls_conf = box[5]
                cls_id = box[6]
                print('%s: %f' % (class_names[cls_id], cls_conf))
                classes = len(class_names)
                offset = cls_id * 123457 % classes
                red = get_color(2, offset, classes)
                green = get_color(1, offset, classes)
                blue = get_color(0, offset, classes)
                if color is None:
                    rgb = (red, green, blue)
                img = cv2.putText(img, class_names[cls_id], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, rgb, 1)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), rgb, 1)
        
        return img


    #
    # Pose estimation is the function that is created by the user.
    #

    def onShutdown(self):
        super(TensorNode, self).onShutdown()


if __name__ == "__main__":
    # Initialize the node
    tensor_node = TensorNode(node_name='tensor_node')
    # Keep it spinning
    rospy.spin()
