import time
import os
import sys

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import torch.nn.functional as F
from cluster import LaneNetCluster
from matplotlib import pyplot as plt

from model import LaneNet

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


def load_test_data(img_path, transform):
    img = Image.open(img_path)
    img = transform(img)
    return img


def test():
    if not os.path.exists('test_output'):
        os.mkdir('test_output')
    if not os.path.exists('temp_img'):
        os.mkdir('temp_img')
    img_path = 'test/test2.jpg'
    video_path = 'test/test.mp4'
    resize_height = 256
    resize_width = 512
    # model_path = 'log/best_model.pth'
    model_path = 'log/epoch_140_model.pth'
    data_transform = transforms.Compose([
        transforms.Resize((resize_height, resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = LaneNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['state_dict'])
    train_loss = state_dict['log']['training_loss']
    val_loss = state_dict['log']['val_loss']
    plt.plot(range(140), train_loss, label='train_loss')
    plt.plot(range(140), val_loss, label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    print(val_loss[130:139])
    # model.eval()
    # model.to(DEVICE)
    # cap = cv2.VideoCapture(video_path)
    # frame_count = 1
    # success = True
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # videoWriter = cv2.VideoWriter('test_output/video_output.mp4', fourcc, 30,
    #                               (resize_width, resize_height))
    # while success:
    #     success, frame = cap.read()
    #     if not success:
    #         break
    #     # params.append(cv.CV_IMWRITE_PXM_BINARY)
    #     # params.append(1)
    #     # cv2.imwrite("temp_img" + "_%d.jpg" % frame_count, frame, params)
    #     fram = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     dummy_input = data_transform(Image.fromarray(frame)).to(DEVICE)
    #     dummy_input = torch.unsqueeze(dummy_input, dim=0)
    #     outputs = model(dummy_input)
    #
    #     input = Image.fromarray(frame)
    #     input = input.resize((resize_width, resize_height))
    #     input = np.array(input)
    #
    #     instance_pred = torch.squeeze(outputs['embed'].detach().to('cpu')).numpy() * 255
    #     # print(outputs['seg'][0][1][np.where(outputs['seg'][0][1] > 0)])
    #     binary_pred = torch.argmax(F.softmax(outputs['seg'], dim=1), dim=1, keepdim=True)
    #     binary_pred = torch.squeeze(binary_pred, 0).to('cpu').numpy() * 255
    #     cluster = LaneNetCluster()
    #     instance_image = cluster.apply_lane_feats_cluster(
    #         binary_seg_result=binary_pred,
    #         instance_seg_result=instance_pred
    #     )
    #     if instance_image is None:
    #         instance_image = np.zeros(shape=(resize_height, resize_width, 3), dtype=np.uint8)
    #     alpha_channel = np.zeros(input.shape[:-1], dtype=np.uint8) + 255
    #     input = cv2.merge((input, alpha_channel))
    #     mask_image = Image.alpha_composite(Image.fromarray(input), Image.fromarray(instance_image))
    #     # mask_image = cv2.addWeighted(input, 0.6, instance_image, 0.4, 0.0)
    #     mask_image = mask_image.convert('RGB')
    #     videoWriter.write(np.array(mask_image))
    #     frame_count = frame_count + 1
    #     if (frame_count % 30) == 0:
    #         print(frame_count)
    # cap.release()
    # videoWriter.release()
    # success, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # dummy_input = data_transform(Image.fromarray(frame)).to(DEVICE)
    # dummy_input = torch.unsqueeze(dummy_input, dim=0)
    # # dummy_input = load_test_data(img_path, data_transform).to(DEVICE)
    # # dummy_input = torch.unsqueeze(dummy_input, dim=0)
    # outputs = model(dummy_input)
    # input = Image.fromarray(frame)
    # input = input.resize((resize_width, resize_height))
    # input = np.array(input)
    # # input = Image.open(img_path)
    # # input = input.resize((resize_width, resize_height))
    # # input = np.array(input)
    #
    # instance_pred = torch.squeeze(outputs['embed'].detach().to('cpu')).numpy() * 255
    # # print(outputs['seg'][0][1][np.where(outputs['seg'][0][1] > 0)])
    # binary_pred = torch.argmax(F.softmax(outputs['seg'], dim=1), dim=1, keepdim=True)
    # binary_pred = torch.squeeze(binary_pred, 0).to('cpu').numpy() * 255
    # cluster = LaneNetCluster()
    # instance_image = cluster.apply_lane_feats_cluster(
    #     binary_seg_result=binary_pred,
    #     instance_seg_result=instance_pred
    # )
    # if instance_image is None:
    #     instance_image = np.zeros(shape=(resize_height, resize_width, 3), dtype=np.uint8)
    # alpha_channel = np.zeros(input.shape[:-1], dtype=np.uint8) + 255
    # input = cv2.merge((input, alpha_channel))
    # mask_image = Image.alpha_composite(Image.fromarray(input), Image.fromarray(instance_image))
    # # mask_image = cv2.addWeighted(input, 0.6, instance_image, 0.4, 0.0)
    # mask_image = mask_image.convert('RGB')
    # mask_image.save(os.path.join('test_output', 'mask_output.jpg'))
    # # cv2.imwrite(os.path.join('test_output', 'mask_output.jpg'), mask_image)
    # cv2.imwrite(os.path.join('test_output', 'instance_output.jpg'), instance_image)
    # cv2.imwrite(os.path.join('test_output', 'embed_output.jpg'), instance_pred.transpose((1, 2, 0)))
    # cv2.imwrite(os.path.join('test_output', 'binary_output.jpg'), binary_pred[0])


if __name__ == "__main__":
    test()
