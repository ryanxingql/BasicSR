import csv
import pyiqa
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            # PyIQA metrics settings
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            pyiqa_metrics = dict()
            for metric_name in list(self.opt['val']['metrics'].keys()):
                if self.opt['val']['metrics'][metric_name]['type'] == 'pyiqa':
                    pyiqa_metric = pyiqa.create_metric(metric_name, device=device)
                    self.opt['val']['metrics'][metric_name][
                        'better'] = 'lower' if pyiqa_metric.lower_better else 'higher'
                    pyiqa_metrics[metric_name] = pyiqa_metric

            # FID settings
            cal_fid = False
            if 'fid' in self.opt['val']['metrics']:
                cal_fid = True
                if self.opt['is_train']:
                    fid_sr_folder = osp.join(self.opt['path']['visualization'], 'fid', 'sr')
                    fid_gt_folder = osp.join(self.opt['path']['visualization'], 'fid', 'gt')
                else:
                    fid_sr_folder = osp.join(self.opt['path']['visualization'], 'fid', dataset_name, 'sr')
                    fid_gt_folder = osp.join(self.opt['path']['visualization'], 'fid', dataset_name, 'gt')

            # initialize metric_results and best_metric_results
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            # zero self.metric_results
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        metric_data_tensor = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            metric_data_tensor['target'] = visuals['result']
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                metric_data_tensor['ref'] = visuals['gt']
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # save images for FID calculation
                if cal_fid:
                    if self.opt['is_train']:
                        save_sr_img_path = osp.join(fid_sr_folder, f'{img_name}.png')
                        save_gt_img_path = osp.join(fid_gt_folder, f'{img_name}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_sr_img_path = osp.join(fid_sr_folder, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                            save_gt_img_path = osp.join(fid_gt_folder, f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_sr_img_path = osp.join(fid_sr_folder, f'{img_name}_{self.opt["name"]}.png')
                            save_gt_img_path = osp.join(fid_gt_folder, f'{img_name}_{self.opt["name"]}.png')
                    imwrite(sr_img, save_sr_img_path)
                    imwrite(gt_img, save_gt_img_path)

                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    if name == 'fid':  # avoid per-sample evaluation
                        continue
                    if name in pyiqa_metrics:
                        self.metric_results[name] += pyiqa_metrics[name](
                            **metric_data_tensor).cpu().item()  # input: img paths or RGB [0,1] tensors; target vs. ref
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            # calculate FID score
            if cal_fid:
                self.metric_results['fid'] = pyiqa_metrics['fid'](target=fid_sr_folder, ref=fid_gt_folder)

            # average and log metric results
            for metric in self.metric_results.keys():
                if metric != 'fid':
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'    # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'    Best: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

        # save csv
        metrics = []
        values = []
        best_values = []
        for metric, value in self.metric_results.items():
            metrics.append(metric)
            values.append(f'{value:.4f}')
            if hasattr(self, 'best_metric_results'):
                best_values.append(f'{self.best_metric_results[dataset_name][metric]["val"]:.4f}')

        csv_path = osp.join(self.opt['path']['results_root'], 'metrics.csv')
        if not osp.exists(csv_path):
            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['iter'] + metrics)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([f'{current_iter}'] + values)
            if hasattr(self, 'best_metric_results'):
                writer.writerow([f'best @ {current_iter}'] + best_values)

        # save Markdown table
        md_path = osp.join(self.opt['path']['results_root'], 'metrics.md')
        if not osp.exists(md_path):
            with open(md_path, mode='w') as f:
                f.write('| iter | ' + ' | '.join(metrics) + ' |\n')
                f.write('| --- | ' + ' | '.join(['---' for _ in metrics]) + ' |\n')
        with open(md_path, mode='a') as f:
            f.write(f'| {current_iter} | ' + ' | '.join(values) + ' |\n')
            if hasattr(self, 'best_metric_results'):
                f.write(f'| best @ {current_iter} | ' + ' | '.join(best_values) + ' |\n')

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
