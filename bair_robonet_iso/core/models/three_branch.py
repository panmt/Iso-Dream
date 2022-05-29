import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_v2 import SpatioTemporalLSTMCell
from core.layers.SpatioTemporalLSTMCell_v2_action import SpatioTemporalLSTMCell_action
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.conv_on_input = self.configs.conv_on_input
        self.res_on_conv = self.configs.res_on_conv
        self.patch_height = configs.img_width // configs.patch_size
        self.patch_width = configs.img_width // configs.patch_size
        self.patch_ch = configs.img_channel * (configs.patch_size ** 2)
        self.action_ch = configs.num_action_ch
        self.rnn_height = self.patch_height
        self.rnn_width = self.patch_width
        self.softmax = nn.Softmax(dim=1)

        if self.configs.conv_on_input == 1:
            self.rnn_height = self.patch_height // 4
            self.rnn_width = self.patch_width // 4
            self.conv_input1 = nn.Conv2d(self.patch_ch, num_hidden[0] // 2,
                                         configs.filter_size,
                                         stride=2, padding=configs.filter_size // 2, bias=False)
            self.conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                         padding=configs.filter_size // 2, bias=False)
            self.encoder_second_free = nn.Conv2d(num_hidden[0], num_hidden[0], configs.filter_size, configs.stride,
                                                 padding=configs.filter_size // 2, bias=False)
            self.encoder_second_action = nn.Conv2d(num_hidden[0], num_hidden[0], configs.filter_size, configs.stride,
                                                   padding=configs.filter_size // 2, bias=False)
            self.action_conv_input1 = nn.Conv2d(self.action_ch, num_hidden[0] // 2, configs.filter_size,
                                                stride=2, padding=configs.filter_size // 2, bias=False)
            self.action_conv_input2 = nn.Conv2d(num_hidden[0] // 2, num_hidden[0], configs.filter_size, stride=2,
                                                padding=configs.filter_size // 2, bias=False)
            self.deconv_output1_free = nn.ConvTranspose2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
            self.deconv_output2_free = nn.ConvTranspose2d(num_hidden[num_layers - 1] // 2, self.patch_ch + 1,
                                                     configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                     bias=False)
            self.deconv_output1_action = nn.ConvTranspose2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1] // 2,
                                                       configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                       bias=False)
            self.deconv_output2_action = nn.ConvTranspose2d(num_hidden[num_layers - 1] // 2, self.patch_ch + 1,
                                                       configs.filter_size, stride=2, padding=configs.filter_size // 2,
                                                       bias=False)

        self.mlp1 = nn.Sequential(
            nn.Linear(num_hidden[0] * 2, num_hidden[0])
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(num_hidden[0], self.action_ch)
        )

        self.conv_action1 = nn.Sequential(
            nn.Conv2d(num_hidden[0], num_hidden[0], configs.filter_size, stride=2,
                      padding=configs.filter_size // 2, bias=False),
            nn.BatchNorm2d(num_hidden[0]),
            nn.Tanh()
        )

        self.conv_action2 = nn.Sequential(
            nn.Conv2d(num_hidden[0], num_hidden[0] * 2, configs.filter_size, stride=2,
                      padding=configs.filter_size // 2, bias=False),
            nn.BatchNorm2d(num_hidden[0] * 2),
            nn.Tanh()
        )

        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4, padding=0)

        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list_free = []
        cell_list_action = []
        self.beta = configs.decouple_beta
        self.MSE_criterion = nn.MSELoss().cuda()

        for i in range(num_layers):
            if i == 0:
                in_channel = self.patch_ch if self.configs.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list_free.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], self.rnn_width,
                                       configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list_free = nn.ModuleList(cell_list_free)

        for i in range(num_layers):
            if i == 0:
                in_channel = num_hidden[i] + self.action_ch if self.configs.conv_on_input == 0 else num_hidden[0]
            else:
                in_channel = num_hidden[i - 1]
            cell_list_action.append(
                SpatioTemporalLSTMCell_action(in_channel, num_hidden[i], self.rnn_width,
                                              configs.filter_size, configs.stride, configs.layer_norm)
            )
        self.cell_list_action = nn.ModuleList(cell_list_action)

        if self.configs.conv_on_input == 0:
            self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.patch_ch + self.action_ch, 1, stride=1,
                                       padding=0, bias=False)
        self.adapter_free = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                 bias=False)

        self.adapter_action = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1], 1, stride=1, padding=0,
                                   bias=False)

    def forward(self, all_frames, mask_true, is_train=True):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = all_frames.permute(0, 1, 4, 2, 3).contiguous()
        input_frames = frames[:, :, :self.patch_ch, :, :]
        input_actions = frames[:, :, self.patch_ch:, :, :]
        # mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]

        next_frames = []
        h_t_free = []
        c_t_free = []
        delta_c_list_free = []
        delta_m_list_free = []

        h_t_action = []
        c_t_action = []
        delta_c_list_action = []
        delta_m_list_action = []

        h_t_test = []
        c_t_test = []
        loss_action = 0

        for i in range(self.num_layers):
            zeros = torch.zeros(
                [self.configs.batch_size, self.num_hidden[i], self.rnn_height, self.rnn_width]).cuda()
            h_t_free.append(zeros)
            c_t_free.append(zeros)
            delta_c_list_free.append(zeros)
            delta_m_list_free.append(zeros)
            h_t_action.append(zeros)
            c_t_action.append(zeros)
            delta_c_list_action.append(zeros)
            delta_m_list_action.append(zeros)
            h_t_test.append(zeros)
            c_t_test.append(zeros)

        decouple_loss_free = []
        decouple_loss_action = []
        memory_free = torch.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()
        memory_action = torch.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()
        memory_test = torch.zeros([self.configs.batch_size, self.num_hidden[0], self.rnn_height, self.rnn_width]).cuda()

        for t in range(self.configs.total_length - 1):
            if t <= 1:
                net = input_frames[:, t]
                net_shape1 = net.size()
                net = self.conv_input1(net)
                net_shape2 = net.size()
                net = self.conv_input2(net)

            else:
                net1 = x_gen_free[:, :3] * mask_free
                net2 = recon_combined

                net_shape1 = net1.size()
                net1 = self.conv_input1(net1)
                net_shape2 = net1.size()
                net1 = self.conv_input2(net1)

                net2 = self.conv_input1(net2)
                net2 = self.conv_input2(net2)

            action = input_actions[:, t]

            if t <= 1:
                net_h1 = self.encoder_second_free(net)
            else:
                net_h1 = self.encoder_second_free(net1)

            h_t_free[0], c_t_free[0], memory_free, delta_c, delta_m = self.cell_list_free[0](net_h1, h_t_free[0], c_t_free[0], memory_free)
            delta_c_list_free[0] = F.normalize(self.adapter_free(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list_free[0] = F.normalize(self.adapter_free(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(1, self.num_layers):
                h_t_free[i], c_t_free[i], memory_free, delta_c, delta_m = self.cell_list_free[i](h_t_free[i - 1], h_t_free[i], c_t_free[i], memory_free)
                delta_c_list_free[i] = F.normalize(self.adapter_free(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
                delta_m_list_free[i] = F.normalize(self.adapter_free(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

            for i in range(0, self.num_layers):
                a = torch.abs(
                    torch.cosine_similarity(delta_c_list_free[i], delta_m_list_free[i], dim=2))
                decouple_loss_free.append(torch.mean(a))
            if self.conv_on_input == 1:
                x_gen_free = self.deconv_output1_free(h_t_free[self.num_layers - 1], output_size=net_shape2)
                x_gen_free = self.deconv_output2_free(x_gen_free, output_size=net_shape1)
            else:
                x_gen_free = self.conv_last(h_t_free[self.num_layers - 1])

            if is_train:
                net_next = self.conv_input1(input_frames[:, t + 1])
                net_next = self.conv_input2(net_next)
                if t <= 1:
                    net_h2 = self.encoder_second_action(net)
                else:
                    net_h2 = self.encoder_second_action(net2)
                net_next = self.encoder_second_action(net_next)
                h2_action = self.conv_action1(net_next - net_h2)
                h2_action = self.conv_action2(h2_action)
                h2_action = self.maxpool(h2_action)
                h_dim = h2_action.shape[1]
                h2_action = h2_action.view(-1, h_dim)
                h2_action = self.mlp1(h2_action)
                ac_recon = self.mlp2(h2_action)
                action_true = action[:, :, 0, 0]
                loss_action += torch.sum((action_true - ac_recon) ** 2) / batch_size

                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)

                h_t_action[0], c_t_action[0], memory_action, delta_c_2, delta_m_2 = self.cell_list_action[0](net_h2, h_t_action[0], c_t_action[0],
                                                                                         memory_action, action)
                delta_c_list_action[0] = F.normalize(
                    self.adapter_action(delta_c_2).view(delta_c_2.shape[0], delta_c_2.shape[1], -1), dim=2)
                delta_m_list_action[0] = F.normalize(
                    self.adapter_action(delta_m_2).view(delta_m_2.shape[0], delta_m_2.shape[1], -1), dim=2)

                for i in range(1, self.num_layers):
                    h_t_action[i], c_t_action[i], memory_action, delta_c_2, delta_m_2 = self.cell_list_action[i](h_t_action[i - 1], h_t_action[i],
                                                                                             c_t_action[i], memory_action, action)
                    delta_c_list_action[i] = F.normalize(
                        self.adapter_action(delta_c_2).view(delta_c_2.shape[0], delta_c_2.shape[1], -1),
                        dim=2)
                    delta_m_list_action[i] = F.normalize(
                        self.adapter_action(delta_m_2).view(delta_m_2.shape[0], delta_m_2.shape[1], -1),
                        dim=2)

                for i in range(0, self.num_layers):
                    b = torch.abs(
                        torch.cosine_similarity(delta_c_list_action[i], delta_m_list_action[i], dim=2))
                    decouple_loss_action.append(torch.mean(b))
                x_gen_action = self.deconv_output1_action(h_t_action[self.num_layers - 1], output_size=net_shape2)
                x_gen_action = self.deconv_output2_action(x_gen_action, output_size=net_shape1)
            else:
                if t <= 1:
                    net_h2 = self.encoder_second_action(net)
                else:
                    net_h2 = self.encoder_second_action(net2)
                action = self.action_conv_input1(action)
                action = self.action_conv_input2(action)
                h_t_test[0], c_t_test[0], memory_test, _, _ = self.cell_list_action[0](net_h2, h_t_test[0], c_t_test[0],
                                                                                  memory_test, action)
                for i in range(1, self.num_layers):
                    h_t_test[i], c_t_test[i], memory_test, _, _ = self.cell_list_action[i](h_t_test[i - 1], h_t_test[i],
                                                                                      c_t_test[i], memory_test, action)

                x_gen_action = self.deconv_output1_action(h_t_test[self.num_layers - 1], output_size=net_shape2)
                x_gen_action = self.deconv_output2_action(x_gen_action, output_size=net_shape1)

            mask_free = torch.sigmoid(x_gen_free[:, -1, :, :]).unsqueeze(1).repeat(1, 3, 1, 1)
            mask_action = torch.sigmoid(x_gen_action[:, -1, :, :]).unsqueeze(1).repeat(1, 3, 1, 1)
            mask_bg = torch.ones_like(mask_action) - mask_free - mask_action

            if t <= 1:
                recon_combined = x_gen_free[:, :3] * mask_free + x_gen_action[:, :3] * mask_action + input_frames[:, t] * mask_bg
            else:
                recon_combined = x_gen_free[:, :3] * mask_free + x_gen_action[:, :3] * mask_action + recon_combined * mask_bg

            next_frames.append(recon_combined)

        if is_train:
            decouple_loss_free = torch.mean(torch.stack(decouple_loss_free, dim=0))
            decouple_loss_action = torch.mean(torch.stack(decouple_loss_action, dim=0))
        else:
            decouple_loss_free = 0
            decouple_loss_action = 0

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # print(next_frames.shape)
        loss_pd = self.MSE_criterion(next_frames,
                                     all_frames[:, 1:, :, :, :next_frames.shape[4]])
        loss = loss_pd + self.beta * decouple_loss_free + self.beta * decouple_loss_action + self.configs.action_beta * loss_action
        next_frames = next_frames[:, :, :, :, :self.patch_ch]
        
        return next_frames, loss, loss_pd, self.configs.action_beta * loss_action

