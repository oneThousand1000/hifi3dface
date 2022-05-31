# -*- coding:utf8 -*-
"""
This file is part of the repo: https://github.com/tencent-ailab/hifi3dface

If you find the code useful, please cite our paper: 

"High-Fidelity 3D Digital Human Head Creation from RGB-D Selfies."
ACM Transactions on Graphics 2021
Code: https://github.com/tencent-ailab/hifi3dface

Copyright (c) [2020-2021] [Tencent AI Lab]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from absl import app, flags
import scipy.io as sio
import cv2
import numpy as np
import sys

sys.path.append("..")
import scipy.ndimage
from RGBD_utils.chooseFrame import chooseFrame
import os


def load_from_npz(base_dir):
    data = np.load(os.path.join(base_dir, "camera_matrix.npz"))
    K = data["K_depth"]

    data = np.load(os.path.join(base_dir, "formatted_data.npz"))
    lmk3d_all = data["lmk3d_all"]
    img_all = data["img_all"]
    depth_all = data["depth_all"]
    rgb_name_all = data["rgb_name_all"]

    temp = np.load(os.path.join(base_dir, "pt3ds.npz"))
    pt3d_all = np.array(temp["pt3d_all"])

    return img_all, depth_all, lmk3d_all, pt3d_all, rgb_name_all, K


def run(prepare_dir, prefit_dir):
    print("---- step1B start -----")
    print("running base:", prepare_dir)

    img_all, depth_all, lmk3d_all, pt3d_all, rgb_name_all, K = load_from_npz(
        prepare_dir
    )

    #####################################################  sort to mid - left - right - up ##############################
    # input 4*2*86 (4 is the number of pics,86 is the number of landmarks)
        
    idmid = 0
    new_list = [idmid]

    lmk3d_select = lmk3d_all[new_list, :, :]
    img_select = img_all[new_list, :, :]
    pt3d_select = pt3d_all[new_list, :, :]
    depth_select = depth_all[new_list, :, :]
    rgb_name_select = rgb_name_all[new_list]

 
    ############################################### calculate pose ################################################
    pt3d_ref = pt3d_select[0].transpose()

    trans_ref = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    first_trans_select = [trans_ref]

    if os.path.exists(prefit_dir) is False:
        os.makedirs(prefit_dir)

    np.savez(
        os.path.join(prefit_dir, "step1_data_for_fusion.npz"),
        first_trans_select=first_trans_select,#
        img_select=img_select,#
        new_list=new_list,
        depth_select=depth_select,
        rgb_name_select=rgb_name_select,
        K=K,
        lmk3d_select=lmk3d_select,
        pt3d_select=pt3d_select,
        inliners=[[0]],
    )

    for i in range(img_select.shape[0]):
        cv2.imwrite(prefit_dir + "/" + str(i) + ".png", img_select[i])

    print("---- step1B succeed -----")


def main(_):
    prepare_dir = FLAGS.prepare_dir
    prefit_dir = FLAGS.prefit
    run(prepare_dir, prefit_dir)


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    flags.DEFINE_string("prepare_dir", "prepare/", "output data directory")
    flags.DEFINE_string("prefit", "prefit_bfm/", "output data directory")

    app.run(main)
