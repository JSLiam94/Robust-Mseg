import numpy as np
import os
import medpy.io as medio
import tensorflow as tf
import shutil

def DeepMedicFormat_crop():
    raw_data_pth = '/home/zhushenghao/data/JS/Brats2024/training_data1_v2'
    save_data_pth = '/home/zhushenghao/data/JS/Robust-Mseg-main/brats_2024'
    data_fds = os.listdir(raw_data_pth)
    data_fds.sort()
    for data_fd in data_fds:
        img_fds = os.listdir(raw_data_pth+'/'+data_fd)
        img_fds.sort()
        for img_fd in img_fds:

            #print(img_fd)
            if not os.path.exists(save_data_pth+'/'+data_fd):
                os.makedirs(save_data_pth+'/'+data_fd)
            modality_fds = os.listdir(raw_data_pth+'/'+data_fd+'/'+img_fd)
            #print(raw_data_pth + '/' + data_fd + '/' + img_fd)
            #print(modality_fds)
            modality_fds.sort()
            modality_fds[1], modality_fds[0] = modality_fds[0], modality_fds[1]

            brainmask_arr_list = []
            for modality_fd in modality_fds:


                if 'seg' not in modality_fd:
                    print("load: "+raw_data_pth + '/' +data_fd+'/'+img_fd + '/' + data_fd+'-'+modality_fd[-10:-7] + '.nii.gz')
                    image_arr, image_header = medio.load(raw_data_pth + '/' +data_fd+'/'+img_fd + '/' + data_fd+'-'+modality_fd[-10:-7] + '.nii.gz')
                    brainmask_arr = image_arr.copy()
                    brainmask_arr[brainmask_arr > 0] = 1
                    brainmask_arr_list.append(brainmask_arr)

            brainmask_arr = brainmask_arr_list[0]
            for m in range(1,len(brainmask_arr_list)):
                brainmask_arr = brainmask_arr+brainmask_arr_list[m]
            brainmask_arr[brainmask_arr>0]=1
            # brainmask_arr[brainmask_arr <4] = 0
            print("save: "+save_data_pth + '/'+data_fd + '/'+ data_fd+'-brainmask.nii.gz')
            medio.save(brainmask_arr,
                       save_data_pth + '/'+data_fd +'/'+data_fd+'-brainmask.nii.gz',
                       image_header)

            brainmask_arr, brainmask_header = medio.load(save_data_pth + '/' +data_fd +'/'+data_fd  +'-brainmask.nii.gz')
            roi_ind = np.where(brainmask_arr > 0)
            roi_bbx = [roi_ind[0].min(), roi_ind[0].max(), roi_ind[1].min(), roi_ind[1].max(), roi_ind[2].min(), roi_ind[2].max()]
            for modality_fd in modality_fds:
                if 'seg' not in modality_fd:

                    image_arr, image_header = medio.load(raw_data_pth + '/' +data_fd+'/'+img_fd + '/' + data_fd+ '-'+modality_fd[-10:-7]+ '.nii.gz')


                    roi_arr = image_arr[brainmask_arr>0]

                    lower_limit = np.percentile(roi_arr, 1)
                    upper_limit = np.percentile(roi_arr, 99)
                    roi_arr = roi_arr[roi_arr>lower_limit]
                    roi_arr = roi_arr[roi_arr<upper_limit]
                    roi_mean = roi_arr.mean()
                    roi_std = roi_arr.std()

                    image_arr = (image_arr-roi_mean)/roi_std
                    image_arr_crop = image_arr[roi_bbx[0]:roi_bbx[1]+1, roi_bbx[2]:roi_bbx[3]+1, roi_bbx[4]:roi_bbx[5]+1]

                    medio.save(image_arr_crop,
                               save_data_pth + '/' +data_fd +  '/' + data_fd+'-'+modality_fd[-10:-7]+'-subtrMeanDivStd' + '.nii.gz')
                    print("save: " + save_data_pth + '/' +data_fd +  '/' + data_fd+'-'+modality_fd[-10:-7]+'-subtrMeanDivStd' + '.nii.gz')
                else:
                    #modality_name = 'OTMultiClass'
                    image_arr, image_header = medio.load(raw_data_pth + '/' +data_fd+'/'+img_fd + '/' + data_fd+'-'+ modality_fd[-10:-7] + '.nii.gz')
                    image_arr_crop = image_arr[roi_bbx[0]:roi_bbx[1] + 1, roi_bbx[2]:roi_bbx[3] + 1, roi_bbx[4]:roi_bbx[5] + 1]
                    medio.save(image_arr_crop,
                               save_data_pth + '/' + data_fd  + '/' +data_fd+'-'+modality_fd[-10:-7] + '.nii.gz',
                               image_header)
                    print("save: " + save_data_pth + '/' + data_fd  + '/'+ data_fd+'-'+modality_fd[-10:-7] + '.nii.gz')

            brainmask_arr_crop = brainmask_arr[roi_bbx[0]:roi_bbx[1]+1, roi_bbx[2]:roi_bbx[3]+1, roi_bbx[4]:roi_bbx[5]+1]
            medio.save(brainmask_arr_crop,
                       save_data_pth + '/' +data_fd  + '/' + data_fd+'-brainmask' + '.nii.gz',
                       brainmask_header)
            print("save: " + save_data_pth + '/' +data_fd  + '/' + data_fd+'-brainmask' + '.nii.gz')


def nii2tfrecord(txt):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    raw_data_pth = '/home/zhushenghao/data/JS/Robust-Mseg-main/brats_2024'
    save_data_pth = '/home/zhushenghao/data/JS/Robust-Mseg-main/brats_all'
    pid_all = os.listdir(raw_data_pth)
    pid_all.sort()

    with open(txt, 'r') as fp:
        rows = fp.readlines()

    pid_all = [row[:-1] for row in rows]
    pid_all.sort()

    cnt = 0
    for pid_indx, pid in enumerate(pid_all):
        cnt +=1
        modality_all = os.listdir(raw_data_pth+'/'+pid)
        modality_all.sort()
        for modality in modality_all:
            data_arr, data_header = medio.load(raw_data_pth + '/' + pid + '/' + modality)
            data_arr = np.float32(data_arr)
            dsize_dim0_val = data_arr.shape[0]
            dsize_dim1_val = data_arr.shape[1]
            dsize_dim2_val = data_arr.shape[2]

            if not os.path.exists(save_data_pth+'/'+pid):
                os.makedirs(save_data_pth+'/'+pid)

            writer = tf.python_io.TFRecordWriter(save_data_pth+'/'+pid+'/'+modality.split('.')[0]+'.tfrecords')

            feature = {'data_vol': tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(data_arr.tostring())])),
                'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim0_val])),
                'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim1_val])),
                'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim2_val])),
                'data_indx': tf.train.Feature(int64_list=tf.train.Int64List(value=[pid_indx])),
                }

            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            writer.close()
            print(pid, modality, cnt)

def nii2tfrecord_all(txt_train,txt_test):
    nii2tfrecord(txt_train)
    nii2tfrecord(txt_test)

if __name__=='__main__':

    txt_train='/home/zhushenghao/data/JS/Robust-Mseg-main/datalist/train.txt'
    txt_test='/home/zhushenghao/data/JS/Robust-Mseg-main/datalist/test.txt'
    #DeepMedicFormat_crop()
    nii2tfrecord_all(txt_train,txt_test)
