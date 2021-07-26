from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils.plots as plt  # noqa
import utils.general as utils  # noqa


class NeuSTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        scan_id = kwargs['scan_id'] if kwargs['scan_id'] != -1 else self.conf.get_int('dataset.scan_id', default=-1)

        if scan_id != -1:
            print("self.expname:{}".format(self.expname))
            self.expname = self.expname + '_{0}'.format(str(scan_id))

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], self.expname)):
                # timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], self.expname))
                timestamps = os.listdir(os.path.join('./', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        self.writer_dir = os.path.join(self.expdir, self.timestamp, 'logs')
        utils.mkdir_ifnotexists(self.writer_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'],
                  os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['scan_id'] != -1:
            dataset_conf['scan_id'] = kwargs['scan_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                                                          **dataset_conf)

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))

        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, self.sched_milestones, gamma=self.sched_factor)

        self.start_epoch = 0
        if is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = saved_model_state['epoch']

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.plot_conf = self.conf.get_config('plot')

        self.writer = SummaryWriter(log_dir=self.writer_dir)
        # self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        # self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        # for acc in self.alpha_milestones:
        #     if self.start_epoch > acc:
        #         self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def run(self):
        print("training...")

        iter_step = 0

        for epoch in range(self.start_epoch, self.nepochs + 1):

            # if epoch in self.alpha_milestones:
            #     self.loss.alpha = self.loss.alpha * self.alpha_factor

            if epoch % 100 == 0:
                self.save_checkpoints(epoch)

            # evaluation
            if epoch % self.plot_freq == 0:
                with torch.no_grad():
                    self.model.eval()

                    self.train_dataset.change_sampling_idx(-1)
                    indices, model_input, ground_truth = next(iter(self.plot_dataloader))

                    model_input["intrinsics"] = model_input["intrinsics"].cuda()
                    model_input["uv"] = model_input["uv"].cuda()

                    model_input["object_mask"] = model_input["object_mask"].cuda()

                    model_input['pose'] = model_input['pose'].cuda()

                    split = utils.split_input(model_input, self.total_pixels)
                    res = []

                    height, width = 1200, 1600
                    # rgb_gt = (ground_truth['rgb'].reshape(3, height, width) + 1.0) / 2.0
                    rgb_gt = (ground_truth['rgb'].reshape(height, width, 3).permute(2, 0, 1) + 1.0) / 2.0
                    self.writer.add_image(
                        "validation/rgb_gt", rgb_gt, epoch
                    )

                    for sp in tqdm(split):
                        out = self.model(sp)
                        res.append({
                            'rgb_values': out['rgb_values'].detach(),
                            'fg_rgb_values': out['fg_rgb_values'].detach(),
                            'bg_rgb_values': out['bg_rgb_values'].detach(),
                        })

                    batch_size = ground_truth['rgb'].shape[0]
                    model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

                    self.writer.add_image(
                        "validation/rgb_prediction", model_outputs['rgb_values'].reshape(
                            height, width, 3).permute(2, 0, 1), epoch
                    )
                    self.writer.add_image(
                        "validation/fg_rgb_prediction", model_outputs['fg_rgb_values'].reshape(
                            height, width, 3).permute(2, 0, 1), epoch
                    )
                    self.writer.add_image(
                        "validation/bg_rgb_prediction", model_outputs['bg_rgb_values'].reshape(
                            height, width, 3).permute(2, 0, 1), epoch
                    )

                    # plt.plot(self.model,
                    #          indices,
                    #          model_outputs,
                    #          model_input['pose'],
                    #          ground_truth['rgb'],
                    #          self.plots_dir,
                    #          epoch,
                    #          self.img_res,
                    #          **self.plot_conf
                    #          )

                    self.model.train()

            self.train_dataset.change_sampling_idx(self.num_pixels)

            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader)):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()

                model_input['pose'] = model_input['pose'].cuda()

                with torch.autograd.detect_anomaly():
                    model_outputs = self.model(model_input)

                    loss_output = self.loss(model_outputs, ground_truth)
                    loss = loss_output['loss']

                    self.optimizer.zero_grad()
                    if self.train_cameras:
                        self.optimizer_cam.zero_grad()

                    loss.backward()

                    self.optimizer.step()

                # print(
                #     '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, s = {7}, lr = {8}'
                #     .format(self.expname, epoch, data_index, self.n_batches, loss.item(),
                #             loss_output['rgb_loss'].item(),
                #             loss_output['eikonal_loss'].item(),
                #             self.model.scale,
                #             self.scheduler.get_lr()[0]))

                self.writer.add_scalar("loss", loss.item(), iter_step)
                self.writer.add_scalar("rgb_loss", loss_output['rgb_loss'].item(), iter_step)
                self.writer.add_scalar("eikonal_loss", loss_output['eikonal_loss'].item(), iter_step)
                self.writer.add_scalar("training_scale_log", self.model.training_scale_log, iter_step)
                self.writer.add_scalar("lr", self.scheduler.get_lr()[0], iter_step)

                iter_step = iter_step + 1

                self.train_dataset.change_sampling_idx(self.num_pixels)

            self.scheduler.step()
