import os
import wandb
from collections import OrderedDict
from importlib import import_module
import json
import cv2
import imageio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from sklearn.metrics import accuracy_score, f1_score, jaccard_score
# from surface_distance import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import largestConnectComponent
from utils.metrics import single_dice_coefficient
from utils.train_utils import make_optimizer
from medpy.metric import dc, jc, sensitivity, hd, hd95, assd
from monai.utils import ensure_tuple_rep
from skimage.segmentation import mark_boundaries


def _get_association(sp, a2, a3, a4, msp, ma2, ma3, ma4, size, sp_num, ran):
    msp *= 0
    # ma1 *= 0
    ma2 *= 0
    ma3 *= 0
    ma4 *= 0

    for i in range(ran):
        msp[i, np.arange(size ** 2), sp[i]] = 1.
        # ma1[i, np.arange(sp_num), a1[i]] = 1
        ma2[i, np.arange(sp_num // 4), a2[i]] = 1.
        ma3[i, np.arange(sp_num // 16), a3[i]] = 1.
        ma4[i, np.arange(sp_num // 64), a4[i]] = 1.

    return msp, ma2, ma3, ma4


class Base_Module(LightningModule):
    def __init__(self, args):
        super(Base_Module, self).__init__()

        self.args = args

    def init_weights(self, pretrained):
        if os.path.isfile(pretrained):
            print("=> loading pretrained model {}".format(pretrained))
            pretrained_dict = torch.load(pretrained, map_location="cpu")
            pretrained_dict = pretrained_dict["state_dict"]
            model_dict = self.state_dict()
            print('Model dict: ', model_dict.keys())
            available_pretrained_dict = {}

            for k, v in pretrained_dict.items():
                print('Pretrained dict: ', k)
                if k in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k].shape:
                        available_pretrained_dict[k] = v
                if k[7:] in model_dict.keys():
                    if pretrained_dict[k].shape == model_dict[k[7:]].shape:
                        available_pretrained_dict[k[7:]] = v

            for k, _ in available_pretrained_dict.items():
                print("loading {}".format(k))
            model_dict.update(available_pretrained_dict)
            self.load_state_dict(model_dict, strict=True)

    def load_weights(self, path):
        if os.path.isfile(path):
            print("=> Loading model from {}".format(path))
            checkpoint = torch.load(path, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "module." in k:
                    name = k[7:]  # remove `module.`
                else:
                    name = k
                print(name)
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            print("=> trained model loaded")

    def initialize(self, decoder, pred):
        if decoder is not None:
            for m in decoder.modules():

                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_uniform_(
                        m.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        if pred is not None:
            for m in pred.modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def configure_optimizers(self):
        optim, kwargs_optimizer = make_optimizer(args=self.args)
        opt = optim(self.parameters(), **kwargs_optimizer)
        scheduler = CosineAnnealingLR(opt, T_max=self.args.epoch)

        return [opt], [scheduler]


class Seg_Model(Base_Module):
    def __init__(self, args, criteria):
        super(Seg_Model, self).__init__(args=args)

        self.criteria = criteria

    def training_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = batch

        output = self(input, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=False)

        if not self.args.dice_loss:
            loss = self.criteria[0](output, target)
        else:
            ce_loss = self.criteria[0](output, target)
            dc_loss = self.criteria[1](output, target)
            jc_loss = self.criteria[2](output, target)
            loss = 1.5 * jc_loss + 1.5 * ce_loss + dc_loss

        self.log("Train/CE Loss", ce_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("Train/DC Loss", dc_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    # def on_train_batch_end(self, outputs, batch, batch_idx, unused=None):
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(name)

    def validation_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, _ = batch

        output = self(input, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=True)

        if not self.args.dice_loss:
            ce_loss = self.criteria[0](output, target)
            dc_loss = 0
        else:
            ce_loss = self.criteria[0](output, target)
            dc_loss = self.criteria[1](output, target)

        dsc = 0
        jac = 0
        for j in range(output.size(0)):
            output_temp = torch.argmax(output[j], dim=0).cpu().detach().numpy()
            output_temp = largestConnectComponent(output_temp)
            target_temp = target[j].cpu().detach().numpy()
            dsc += dc(output_temp, target_temp)
            jac += jc(output_temp, target_temp)

        if self.args.use_tensorboard:
            tensorboard = self.logger.experiment
            tensorboard.add_image('Val/Input', input, self.current_epoch, dataformats='NCHW')
            tensorboard.add_image('Val/Output', torch.argmax(output, 1).unsqueeze(1), self.current_epoch, dataformats='NCHW')
            tensorboard.add_image('Val/GT', target.unsqueeze(1), self.current_epoch, dataformats='NCHW')

        self.log("Val/CE Loss", ce_loss, on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/DC Loss", dc_loss, on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/Dice", dsc / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/Jac", jac / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)

        return dsc

    def on_test_start(self):
        self.names = []
        self.dscs = []
        self.jacs = []
        self.accs = []
        self.sens = []
        self.HDs = []
        self.HD95s = []
        self.ASSD = []

    def test_step(self, batch, batch_idx):
        print('=> Testing')
        dirname = "{}/results_{}".format(self.args.save_name, self.args.dataset_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        dsc_sum = 0
        jac_sum = 0
        acc_sum = 0
        sens_sum = 0
        HDs_sum = 0
        HD95s_sum = 0
        ASSD_sum = 0

        images, targets_subset, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, names_subset = batch

        output = self(images, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=True)

        for idx, name in enumerate(names_subset):
            output_np = torch.argmax(output[idx], dim=0).cpu().detach().numpy()
            binary_output = np.array(output_np)
            binary_output = largestConnectComponent(binary_output)
            target_np = targets_subset[idx].cpu().detach().numpy().astype(np.uint8)
            if np.amax(target_np) == 0:
                # change the value of center pixel of target_np to 1
                target_np[target_np.shape[0] // 2, target_np.shape[1] // 2] = 1
            if self.args.save_results:
                filename = os.path.join(dirname, str(name) + ".png")
                image_save = images[idx].permute(1, 2, 0).cpu().detach().numpy()
                image_save = ((image_save * 0.5 + 0.5) * 255).astype(np.uint8)
                # target_save = (target_np * 255).astype(np.uint8)
                # # Change target_save to 3 channels
                # target_save = np.stack((target_save, target_save, target_save), axis=2)
                # output_save = (binary_output * 255).astype(np.uint8)
                # # Change output_save to 3 channels
                # output_save = np.stack((output_save, output_save, output_save), axis=2)
                # # Concatenate image, target, and output to save as an image
                # imageio.imwrite(filename, np.concatenate((image_save, target_save, output_save), axis=1))
                # # imageio.imwrite(filename, (binary_output * 255).astype(np.uint8))

                dir_boundary = "{}_boundary/".format(dirname)
                print(dir_boundary)
                if not os.path.exists(dir_boundary):
                    os.makedirs(dir_boundary)
                filename_boundary = os.path.join(dir_boundary, str(name) + ".png")
                output_boundary = mark_boundaries(image_save, binary_output, color=(0, 1, 1), outline_color=None, mode='thick')
                imageio.imwrite(filename_boundary, (output_boundary * 255).astype(np.uint8))
                print('=> Saved boundary image')

            dsc = dc(binary_output, target_np)
            jac = jc(binary_output, target_np)
            sens = sensitivity(binary_output, target_np)
            haus = hd(binary_output, target_np)
            haus95 = hd95(binary_output, target_np)
            asd = assd(binary_output, target_np)

            target_1d = np.reshape(target_np, (-1, 1))
            pred_1d = np.reshape(binary_output, (-1, 1))
            accuracy = accuracy_score(target_1d, pred_1d)

            self.names.append(name)
            dsc_sum += dsc
            self.dscs.append(dsc)
            jac_sum += jac
            self.jacs.append(jac)
            acc_sum += accuracy
            self.accs.append(accuracy)
            sens_sum += sens
            self.sens.append(sens)
            HDs_sum += haus
            self.HDs.append(haus)
            HD95s_sum += haus95
            self.HD95s.append(haus95)
            ASSD_sum += asd
            self.ASSD.append(asd)

            self.log("Test Dice", dsc, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test Jac", jac, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test Acc", accuracy, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test Sens", sens, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test HDs", haus, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test HD95s", haus95, on_step=False, on_epoch=True, sync_dist=True)
            self.log("Test ASSD", asd, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_end(self):
        print('=> Writing results to local file')
        dataframe = pd.DataFrame(
            {
                "name": self.names,
                "dice": self.dscs,
                "jac": self.jacs,
                "acc": self.accs,
                "sen": self.sens,
                "HD": self.HDs,
                "HD95": self.HD95s,
                "ASSD": self.ASSD
            }
        )
        dataframe.to_excel(
            os.path.join(
                self.args.save_name,
                "count_results_{}_{}.xlsx".format(
                    self.args.dataset_name, self.args.fold
                ),
            )
        )
        print('=> Dataframe saved')
        results = {
            "Dsc": np.average(self.dscs),
            "Jac": np.average(self.jacs),
            "ACC": np.average(self.accs),
            "Sen": np.average(self.sens),
            "HD": np.average(self.HDs),
            "HD95": np.average(self.HD95s),
            "ASSD": np.average(self.ASSD)
        }
        results_json = json.dumps(results, indent=4)
        with open(os.path.join(self.args.save_name, 'results_{}_{}.json'.format(self.args.dataset_name, self.args.fold)), 'w') as f:
            f.write(results_json)
        print('=> Json result saved')


class Ensemble_Seg_Model(Seg_Model):
    def __init__(self, args, criteria):
        super(Ensemble_Seg_Model, self).__init__(args=args, criteria=criteria)

    def validation_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, _ = batch

        # Soft voting
        # output = None
        # for i in range(matrix_sp.size(dim=1)):
        #     output_ = self(input, matrix_sp[:, i], matrix_a1[:, i], matrix_a2[:, i], matrix_a3[:, i], matrix_a4[:, i], inference=True)
        #     if i == 0:
        #         output = output_
        #     else:
        #         output = output + output_
        # output = output / matrix_sp.size(dim=1)

        # Hard voting
        outputs = []
        for i in range(matrix_sp.size(dim=1)):
            output_ = self(input, matrix_sp[:, i], matrix_a1[:, i], matrix_a2[:, i], matrix_a3[:, i], matrix_a4[:, i], inference=True)
            outputs.append(output_)
        for j, output in enumerate(outputs):
            output = output.squeeze()
            output_temp = torch.argmax(output, dim=1)
            if j == 0:
                outputs_ensemble = output_temp
            else:
                outputs_ensemble += output_temp
        outputs_ensemble = torch.where(outputs_ensemble >= (matrix_sp.size(dim=1) // 2), 1, 0)

        if not self.args.dice_loss:
            ce_loss = self.criteria[0](output, target)
            dc_loss = 0
        else:
            ce_loss = self.criteria[0](output, target)
            dc_loss = self.criteria[1](output, target)
            jc_loss = self.criteria[2](output, target)

        dsc_soft = 0
        jac_soft = 0
        dsc_hard = 0
        jac_hard = 0
        for j in range(output.size(0)):
            # output_temp_soft = torch.argmax(output[j], dim=0).cpu().detach().numpy()
            # output_temp_soft = largestConnectComponent(output_temp_soft)
            output_temp_hard = outputs_ensemble[j].cpu().detach().numpy()
            output_temp_hard = largestConnectComponent(output_temp_hard)
            target_temp = target[j].cpu().detach().numpy()
            # dsc_soft += dc(output_temp_soft, target_temp)
            # jac_soft += jc(output_temp_soft, target_temp)
            dsc_hard += dc(output_temp_hard, target_temp)
            jac_hard += jc(output_temp_hard, target_temp)

        # if jac_soft > jac_hard:
        # dsc = dsc_soft
        # jac = jac_soft
        # else:
        dsc = dsc_hard
        jac = jac_hard

        if self.args.use_tensorboard:
            tensorboard = self.logger.experiment
            tensorboard.add_image('Val/Input', input, self.current_epoch, dataformats='NCHW')
            tensorboard.add_image('Val/Output', torch.argmax(output, 1).unsqueeze(1), self.current_epoch, dataformats='NCHW')
            tensorboard.add_image('Val/GT', target.unsqueeze(1), self.current_epoch, dataformats='NCHW')

        self.log("Val/CE Loss", ce_loss, on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/DC Loss", dc_loss, on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/Dice", dsc / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)
        self.log("Val/Jac", jac / output.size(0), on_epoch=True, sync_dist=True, on_step=False, prog_bar=True)

        return dsc

    def on_test_start(self):
        self.names = []
        self.dscs = []
        self.jacs = []
        self.accs = []
        self.sens = []
        self.HDs = []
        self.HD95s = []
        self.ASSD = []

    def test_step(self, batch, batch_idx):
        dirname = "{}/results_{}".format(self.args.save_name, self.args.dataset_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        dsc_sum = 0
        jac_sum = 0
        acc_sum = 0
        sens_sum = 0
        HDs_sum = 0
        HD95s_sum = 0
        ASSD_sum = 0

        images, targets_subset, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, names_subset = batch

        # Soft voting
        # output = None
        # for i in range(matrix_sp.size(dim=1)):
        #     output_ = self(images, matrix_sp[:, i], matrix_a1[:, i], matrix_a2[:, i], matrix_a3[:, i], matrix_a4[:, i], inference=True)
        #     if i == 0:
        #         output = output_
        #     else:
        #         output = output + output_
        # output = output / matrix_sp.size(dim=1)

        # Hard voting
        outputs = []
        output_subs = []
        for i in range(matrix_sp.size(dim=1)):
            output_ = self(images, matrix_sp[:, i], matrix_a1[:, i], matrix_a2[:, i], matrix_a3[:, i], matrix_a4[:, i], inference=True)
            outputs.append(output_)
        for j, output_sub in enumerate(outputs):
            output_sub = output_sub.squeeze()
            output_temp = torch.argmax(output_sub, dim=1)
            output_subs.append(output_temp)
            if j == 0:
                outputs_ensemble = output_temp
            else:
                outputs_ensemble += output_temp
        output = torch.where(outputs_ensemble >= (matrix_sp.size(dim=1) // 2), 1, 0)

        # output = torch.argmax(output, dim=1)
        output = output.squeeze()

        for idx, name in enumerate(names_subset):
            output_np = output[idx].cpu().detach().numpy()
            binary_output = largestConnectComponent(np.array(output_np))
            target_np = targets_subset[idx].cpu().detach().numpy()
            if np.amax(target_np) == 0:
                # change the value of center pixel of target_np to 1
                target_np[target_np.shape[0] // 2, target_np.shape[1] // 2] = 1
            if self.args.save_results:
                # filename0 = os.path.join(dirname, str(self.args.stage[i]))
                # if not os.path.exists(filename0):
                #     os.makedirs(filename0)
                filename = os.path.join(dirname, str(name) + ".png")
                if isinstance(images, list):
                    images = images[0]
                image_save = images[idx].permute(1, 2, 0).cpu().detach().numpy()
                image_save = ((image_save * 0.5 + 0.5) * 255).astype(np.uint8)
                target_save = (target_np * 255).astype(np.uint8)
                # Change target_save to 3 channels
                target_save = np.stack((target_save, target_save, target_save), axis=2)

                # Save color and all results
                # output_saves = None
                # output_color = np.zeros_like(binary_output, dtype=np.uint8)
                # for k, out in enumerate(output_subs):
                #     output_save = largestConnectComponent(out[idx].cpu().detach().numpy())
                #     output_color += output_save.copy().astype(np.uint8)
                #     output_save = (output_save * 255).astype(np.uint8)
                #     # Change output_save to 3 channels
                #     output_save = np.stack((output_save, output_save, output_save), axis=2)
                #     if k == 0:
                #         output_saves = output_save
                #     else:
                #         output_saves = np.concatenate((output_saves, output_save), axis=1)
                # # output_color = outputs_ensemble[idx].cpu().detach().numpy()
                # output_color = ((output_color / np.amax(output_color)) * 255).astype(np.uint8)
                # output_color = cv2.applyColorMap(output_color, cv2.COLORMAP_JET)[..., ::-1]
                # # Concatenate image, target, and output to save as an image
                # output_save = (binary_output * 255).astype(np.uint8)
                # # # Change output_save to 3 channels
                # output_save = np.stack((output_save, output_save, output_save), axis=2)
                # # Concatenate image, target, and output to save as an image
                # # imageio.imwrite(filename, np.concatenate((image_save, target_save, output_save), axis=1))
                # imageio.imwrite(filename, np.concatenate((image_save, target_save, output_color, output_save, output_saves), axis=1))
                # # imageio.imwrite(filename, (binary_output * 255).astype(np.uint8))

                dir_boundary = "{}_boundary/".format(dirname)
                if not os.path.exists(dir_boundary):
                    os.makedirs(dir_boundary)
                filename_boundary = os.path.join(dir_boundary, str(name) + ".png")
                output_boundary = mark_boundaries(image_save, binary_output, color=(0, 1, 1), outline_color=None, mode='thick')
                imageio.imwrite(filename_boundary, (output_boundary * 255).astype(np.uint8))
                print('=> Saved boundary image')

            dsc = dc(binary_output, target_np)
            jac = jc(binary_output, target_np)
            sens = sensitivity(binary_output, target_np)
            haus = hd(binary_output, target_np)
            haus95 = hd95(binary_output, target_np)
            asd = assd(binary_output, target_np)

            target_1d = np.reshape(target_np, (-1, 1))
            pred_1d = np.reshape(binary_output, (-1, 1))
            accuracy = accuracy_score(target_1d, pred_1d)

            if jac > 0.0:
                self.names.append(name)
                dsc_sum += dsc
                self.dscs.append(dsc)
                jac_sum += jac
                self.jacs.append(jac)
                acc_sum += accuracy
                self.accs.append(accuracy)
                sens_sum += sens
                self.sens.append(sens)
                HDs_sum += haus
                self.HDs.append(haus)
                HD95s_sum += haus95
                self.HD95s.append(haus95)
                ASSD_sum += asd
                self.ASSD.append(asd)

                self.log("Test/Dice", dsc, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/Jac", jac, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/Acc", accuracy, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/Sens", sens, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/HDs", haus, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/HD95s", haus95, on_step=False, on_epoch=True, sync_dist=True)
                self.log("Test/ASSD", asd, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_end(self):
        print('=> Writing results to local file')
        dataframe = pd.DataFrame(
            {
                "name": self.names,
                "dice": self.dscs,
                "jac": self.jacs,
                "acc": self.accs,
                "sen": self.sens,
                "HD": self.HDs,
                "HD95": self.HD95s,
                "ASSD": self.ASSD
            }
        )
        dataframe.to_excel(
            os.path.join(
                self.args.save_name,
                "count_results_{}_{}.xlsx".format(
                    self.args.dataset_name, self.args.fold
                ),
            )
        )
        print('=> Dataframe saved')
        results = {
            "Dsc": np.average(self.dscs),
            "Jac": np.average(self.jacs),
            "ACC": np.average(self.accs),
            "Sen": np.average(self.sens),
            "HD": np.average(self.HDs),
            "HD95": np.average(self.HD95s),
            "ASSD": np.average(self.ASSD)
        }
        results_json = json.dumps(results, indent=4)
        with open(os.path.join(self.args.save_name, 'results_{}_{}_{}.json'.format(self.args.dataset_name, self.args.fold, self.args.seed)), 'w') as f:
            f.write(results_json)
        print('=> Json result saved')


class Context_Model(Base_Module):
    def __init__(self, args, criteria):
        super(Context_Model, self).__init__(args=args)

        self.criteria = criteria

    def training_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4 = batch

        output = self(input, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4)

        loss = self.criteria(output, target)
        loss = loss * 100
        self.log(
            "Inpainting Train Loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, _ = batch

        output = self(input, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=True)

        loss = self.criteria(output, target)
        loss = loss * 100
        self.log(
            "Val/InpLoss",
            loss,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        images, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, names_subset = batch

        dirname = "{}/results_inpainting".format(self.args.save_name)
        output = self(images, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=True)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # loss = self.criteria(output, target)

        if self.args.save_results:
            for idx, name in enumerate(names_subset):
                image_np = images.cpu()[idx][:3].permute(1, 2, 0).numpy()
                image_rgb = np.array(image_np) * 0.5 + 0.5
                target_np = target.cpu()[idx].permute(1, 2, 0).numpy()
                target_rgb = np.array(target_np) * 0.5 + 0.5
                output_np = output.cpu()[idx].permute(1, 2, 0).numpy()
                output_rgb = np.array(output_np) * 0.5 + 0.5
                filename = os.path.join(dirname, str(name) + ".png")
                output_stack = np.concatenate((image_rgb + target_rgb, output_rgb), axis=1)
                imageio.imwrite(
                    filename, (output_stack * 255).astype(np.uint8)
                )

        # self.log("Test Loss", loss)


class Ensemble_Context_Model(Context_Model):
    def __init__(self, args, criteria):
        super(Ensemble_Context_Model, self).__init__(args=args, criteria=criteria)

    def validation_step(self, batch, batch_idx):
        input, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, _ = batch

        outputs = []
        for i in range(matrix_sp.size(dim=1)):
            output_ = self(input, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, inference=True)
            outputs.append(output_)

        for i, output in enumerate(outputs):
            if i == 0:
                loss = self.criteria(output, target[i])
            else:
                loss += self.criteria(output, target[i])
        loss = loss * 10

        self.log(
            "Val/InpLoss",
            loss,
            on_epoch=True,
            sync_dist=True,
            on_step=False,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        images, target, matrix_sp, matrix_a1, matrix_a2, matrix_a3, matrix_a4, names_subset = batch

        for i in range(matrix_sp.size(dim=1)):
            dirname = "{}/results_inpainting_{}".format(self.args.save_name, i)
            output = self(images, matrix_sp[:, i], matrix_a1[:, i], matrix_a2[:, i], matrix_a3[:, i], matrix_a4[:, i], inference=True)

            if not os.path.exists(dirname):
                os.makedirs(dirname)

            if self.args.save_results:
                for idx, name in enumerate(names_subset):
                    image_np = images.cpu()[idx][:3].permute(1, 2, 0).numpy()
                    image_rgb = np.array(image_np) * 0.5 + 0.5
                    target_np = target.cpu()[idx].permute(1, 2, 0).numpy()
                    target_rgb = np.array(target_np) * 0.5 + 0.5
                    output_np = output.cpu()[idx].permute(1, 2, 0).numpy()
                    output_rgb = np.array(output_np) * 0.5 + 0.5
                    filename = os.path.join(dirname, str(name) + ".png")
                    output_stack = np.concatenate((image_rgb + target_rgb, image_rgb + output_rgb), axis=1)
                    imageio.imwrite(
                        filename, (output_stack * 255).astype(np.uint8)
                    )


def load_model(criteria, args):
    print("=> creating model '{}'".format(args.model))
    m = import_module("model." + args.model.lower())
    model_name = args.model + "_context" if args.inpainting else args.model
    if args.transformer:
        if args.model == "UNETR":
            model = getattr(m, model_name)(
                args,
                input_channel=3,
                criteria=criteria,
                img_size=args.size,
            )
        else:
            model = getattr(m, model_name)(
                args,
                input_channel=3,
                criteria=criteria,
                img_size=args.size,
                patch_res=args.patch_size,
                window_res=args.window_size,
                feature_size=args.embed_dim,
                depths=args.depths,
                num_heads=args.num_heads,
            )
    else:
        model = getattr(m, model_name)(
            args,
            criteria=criteria,
        )
    if os.path.isfile(args.pretrained):
        print("Pretrained:", args.pretrained)
        model.init_weights(args.pretrained)

    return model
