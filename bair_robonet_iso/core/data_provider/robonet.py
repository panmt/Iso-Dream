import numpy as np
import os
import cv2
from PIL import Image
import logging
import random
import pickle

logger = logging.getLogger(__name__)


class InputHandle:
    def __init__(self, datas, indices, action, input_param, mode):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        if mode == 'train':
            self.minibatch_size = input_param['minibatch_size']
        else:
            self.minibatch_size = input_param['minibatch_size_test']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.action = action
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.add_ball = input_param['add_ball']

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None

        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width, 8)).astype(
            self.input_data_type)

        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            action_slice = self.action[begin:end, :]
            
            if self.add_ball:
                width = 64
                height = 64
                r = 3
                x1 = random.randint(r, width - r)
                y1 = random.randint(r, height - r)
                speed_x1 = 4
                speed_y1 = 4
                x2 = random.randint(r, width - r)
                y2 = random.randint(r, height - r)
                speed_x2 = 4
                speed_y2 = 4
                x3 = random.randint(r, width - r)
                y3 = random.randint(r, height - r)
                speed_x3 = 4
                speed_y3 = 4
            for k in range(action_slice.shape[0]):
                data_image = data_slice[k, :, :, :].copy()
                if self.add_ball:
                    cv2.circle(data_image, (x1, y1), r, (0, 0, 1), -1)
                    cv2.circle(data_image, (x2, y2), r, (0, 1, 0), -1)
                    cv2.circle(data_image, (x3, y3), r, (1, 0, 0), -1)

                    x1 = x1 + speed_x1
                    y1 = y1 + speed_y1
                    if x1 >= width - r or x1 <= r:
                        speed_x1 = -speed_x1
                    if y1 >= height - r or y1 <= r:
                        speed_y1 = -speed_y1

                    x2 = x2 + speed_x2
                    y2 = y2 + speed_y2
                    if x2 >= width - r or x2 <= r:
                        speed_x2 = -speed_x2
                    if y2 >= height - r or y2 <= r:
                        speed_y2 = -speed_y2

                    x3 = x3 + speed_x3
                    y3 = y3 + speed_y3
                    if x3 >= width - r or x3 <= r:
                        speed_x3 = -speed_x3
                    if y3 >= height - r or y3 <= r:
                        speed_y3 = -speed_y3

                input_batch[i, k, :, :, :3] = data_image
                input_batch[i, k, :, :, 3:] = np.stack([np.ones([64, 64]) * i for i in action_slice[k, :]], axis=2)

        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))


class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.category = ['berkeley', 'google', 'penn', 'stanford']
        self.image_width = input_param['image_width']
        self.train_4800 = 3840
        self.train_7800 = 6240

        self.input_param = input_param
        self.seq_len = input_param['seq_length']

    # Specific_category is for Continual learning for different categories
    def load_data(self, paths, mode='train', specific_category=None):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        print('begin load data' + str(path))

        frames_np = []
        frames_file_name = []
        frames_person_mark = []
        frames_category = []
        person_mark = 0
        actions_np = []

        zero_action = np.zeros((1, 5))

        c_dir_list = self.category

        frame_category_flag = -1
        for c_dir in c_dir_list:  

            person_id = []
            if c_dir != 'google':
                if mode == 'train':
                    for j in range(self.train_4800):
                        tmp = '%04d' % j
                        person_id.append(tmp)
                elif mode == 'test':
                    for j in range(self.train_4800, 4800):
                        tmp = '%04d' % j
                        person_id.append(tmp)
                else:
                    print("ERROR!")
            else:
                if mode == 'train':
                    for j in range(self.train_7800):
                        tmp = '%04d' % j
                        person_id.append(tmp)
                elif mode == 'test':
                    for j in range(self.train_7800, 7800):
                        tmp = '%04d' % j
                        person_id.append(tmp)
                else:
                    print("ERROR!")

            if mode == 'train':
                frame_category_flag = 2
            else:
                frame_category_flag = 1

            # load action data
            action_f = open('/data/robonet_environ2/' + c_dir + '.pkl', 'rb')
            action_data = pickle.load(action_f)

            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)

            for p_c_dir in p_c_dir_list:
                if p_c_dir not in person_id:
                    continue

                # read action
                p_c_dir_index = int(p_c_dir)
                action_sequence = action_data[p_c_dir_index]
                if action_sequence.shape[1] == 4:
                    zeros_pad = np.zeros((action_sequence.shape[0], 1))
                    action_sequence = np.concatenate((action_sequence, zeros_pad), 1)
                action_sequence = np.concatenate((action_sequence, zero_action), 0)
                actions_np.append(action_sequence)

                person_mark += 1
                dir_path = os.path.join(c_dir_path, p_c_dir)
                filelist = os.listdir(dir_path)
                filelist.sort()
                for file in filelist:
                    frame_im = Image.open(os.path.join(dir_path, file))
                    frame_np = np.array(frame_im)  
                    frames_np.append(frame_np)
                    frames_file_name.append(file)
                    frames_person_mark.append(person_mark)
                    frames_category.append(frame_category_flag)

        # is it a begin index of sequence
        indices = []
        index = len(frames_person_mark) - 1
        while index >= self.seq_len - 1:
            if frames_person_mark[index] == frames_person_mark[index - self.seq_len + 1]:
                end = int(frames_file_name[index][0:4])
                start = int(frames_file_name[index - self.seq_len + 1][0:4])
                # TODO: mode == 'test'
                if end - start == self.seq_len - 1:
                    indices.append(index - self.seq_len + 1)
                    if frames_category[index] == 1:
                        index -= self.seq_len - 1
                    elif frames_category[index] == 2:
                        index -= 2
                    else:
                        print("category error 2 !!!")
            index -= 1

        actions_np = np.concatenate(actions_np, 0)
        frames_np = np.asarray(frames_np)
        data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, 3))
        action_data = np.zeros((frames_np.shape[0], 5))
        for i in range(len(frames_np)):
            temp = np.float32(frames_np[i, :, :, :])
            data[i, :, :, :] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
            action_data[i, :] = actions_np[i, :]
        print("there are " + str(data.shape[0]) + " pictures")
        print("there are " + str(len(indices)) + " sequences")
        print("there are " + str(action_data.shape[0]) + "actions")
        return data, indices, action_data

    def get_train_input_handle(self, specific_category=None):
        train_data, train_indices, train_action = self.load_data(self.paths, mode='train',
                                                                 specific_category=specific_category)
        return InputHandle(train_data, train_indices, train_action, self.input_param, mode='train')

    def get_test_input_handle(self, specific_category=None):
        test_data, test_indices, test_action = self.load_data(self.paths, mode='test',
                                                              specific_category=specific_category)
        return InputHandle(test_data, test_indices, test_action, self.input_param, mode='test')


