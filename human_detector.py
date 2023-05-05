from models import *
from utils import *
import os
from torchvision import datasets, transforms
class DetectHumans:

    """Summary
    The Detection class

    Attributes:
        class_path (str): Description
        classes (TYPE): Description
        conf_thres (float): Description
        config_path (str): Description
        detectionFlags (list): Description
        f_height (TYPE): Description
        f_width (TYPE): Description
        img_size (int): Description
        input_video_path (TYPE): Description
        model (TYPE): Description
        nms_thres (float): Description
        ouput_video_path (TYPE): Description
        Tensor (TYPE): Description
        weights_path (str): Description
    """

    def __init__(self):

        self.config_path='config/yolov3.cfg'
        self.weights_path='config/yolov3.weights'
        self.class_path='config/coco.names'

        self.img_size=416
        self.conf_thres=0.5
        self.nms_thres=0.4
        self.detectionFlags=[]
        self.load_model()

    def load_model(self):
        """Summary
        Load model and weights
        """

        self.model = Darknet(self.config_path, img_size=self.img_size)

        #check if yolov3.weights file exists else download it
        if not os.path.exists(self.weights_path):
            print("downloading weights from web")
            filename=self.weights_path
            url="https://pjreddie.com/media/files/yolov3.weights"
            chunkSize = 1024
            r = requests.get(url, stream=True)
            with open(filename, 'wb') as f:
                pbar = tqdm( unit="B", total=int( r.headers['Content-Length'] ) )
                for chunk in r.iter_content(chunk_size=chunkSize):
                    if chunk: # filter out keep-alive new chunks
                        pbar.update (len(chunk))
                        f.write(chunk)



        self.model.load_weights(self.weights_path)
        self.model.to("cpu")
        self.model.eval()
        self.classes = utils.load_classes(self.class_path)
        self.Tensor = torch.FloatTensor

    def detect_image(self,img):
        """Summary
        Detect Humans
        Args:
            img (Image): the current frame to process

        Returns:
            all detected objects
        """
        pred_img = Image.fromarray(img)
        # scale and pad image
        ratio = min(self.img_size/pred_img.size[0], self.img_size/pred_img.size[1])
        imw = round(pred_img.size[0] * ratio)
        imh = round(pred_img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
                                              transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                                                             (128,128,128)),
                                              transforms.ToTensor(),
                                              ])

        # convert image to Tensor
        image_tensor = img_transforms(pred_img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = image_tensor.type(self.Tensor)

        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, self.conf_thres, self.nms_thres)

        rectangles = []
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:
            if cls_pred==0:
                pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
                pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
                unpad_h = self.img_size - pad_y
                unpad_w = self.img_size - pad_x

                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                rectangles.append(
                    {
                        "x1" : int(x1),
                        "y1" : int(y1),
                        "x2" : int(x1+box_w),
                        "y2" : int(y1+box_h)
                    }
                )
        return rectangles