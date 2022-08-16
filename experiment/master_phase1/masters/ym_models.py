from strategy_integration.components.models.deep_models.base_model import BaseModel
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, NoReturn


def tensor_nanmean(x: torch.Tensor, dim, keepdim):
    if dim is not None:
        num_finite = torch.isfinite(x).sum(dim=dim, keepdim=keepdim)
        sum_finite = torch.nansum(x, dim=dim, keepdim=keepdim)
    else:
        num_finite = torch.isfinite(x).sum()
        sum_finite = torch.nansum(x)

    return torch.where(
        num_finite != 0, sum_finite / num_finite, torch.zeros_like(sum_finite)
    )


def tensor_nan_z(x: torch.Tensor, dim):
    mu_x = tensor_nanmean(x, dim, True)
    mu_x2 = tensor_nanmean(torch.square(x), dim, True)
    std_x = torch.sqrt(mu_x2 - torch.square(mu_x)) + 1e-6

    return (x - mu_x) / std_x


def tensor_z(x: torch.Tensor, dim):
    mu_x = x.mean(dim=dim, keepdim=True)
    std_x = x.std(dim=dim, keepdim=True) + 1e-6

    return (x - mu_x) / std_x


class InitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self._linear = nn.Linear(in_features, out_features, bias)
        nn.init.kaiming_uniform_(self._linear.weight, a=math.sqrt(8))

        if self._linear.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self._linear.weight)
            bound = math.sqrt(0.5) / math.sqrt(fan_in)
            nn.init.uniform_(self._linear.bias, -bound, bound)

    def forward(self, x):
        return self._linear(x)


class Attention(nn.Module):
    def __init__(
        self, in_features, out_features, dropout=0.0, layer_norm=False, kind="sample"
    ):
        nn.Module.__init__(self)
        assert kind in ("sample", "batch")

        self._in_features = in_features
        self._out_features = out_features
        self._dropout = dropout
        self._kind = kind

        self.q_fc = InitLinear(in_features, out_features, bias=False)
        self.k_fc = InitLinear(in_features, out_features, bias=False)
        self.v_fc = InitLinear(in_features, out_features, bias=False)
        self.o_fc = InitLinear(out_features, out_features, bias=False)
        self.fc1 = InitLinear(out_features, out_features, bias=True)
        self.fc2 = InitLinear(out_features, out_features, bias=True)

        if dropout <= 1e-6:
            self.attn_dropout = nn.Identity()
            self.fc_dropout = nn.Identity()
        else:
            self.attn_dropout = nn.Dropout(dropout)
            self.fc_dropout = nn.Dropout(dropout)

        if layer_norm:
            self.attn_layer_norm = nn.LayerNorm(out_features)
            self.fc_layer_norm = nn.LayerNorm(out_features)
        else:
            self.attn_layer_norm = nn.Identity()
            self.fc_layer_norm = nn.Identity()

    def forward(self, x: torch.Tensor):
        if self._kind == "sample":
            if x.ndim != 4:
                raise ValueError
            dim0, dim1 = 2, 3
        else:
            if x.ndim != 3:
                raise ValueError
            dim0, dim1 = 1, 2

        q = self.q_fc(x)
        k = self.k_fc(x)
        v = self.v_fc(x)

        attn = torch.matmul(q, torch.transpose(k, dim0, dim1))
        attn = F.softmax(attn / self._out_features ** 0.5, dim=dim1)
        v = torch.matmul(attn, v)

        attn = self.attn_dropout(self.o_fc(v))
        attn = self.attn_layer_norm(attn + q)

        out = self.fc2(F.leaky_relu(self.fc1(attn), negative_slope=0.2))
        out = self.fc_dropout(out)
        out = self.fc_layer_norm(out + attn)

        return out


class SampleImputation(nn.Module):
    def __init__(self, sample_shape, noise=0.1):
        nn.Module.__init__(self)
        self.sample_shape = sample_shape
        self.noise = noise

        time_length, cross_length = sample_shape
        self.weight = nn.Parameter(torch.Tensor(time_length, time_length, cross_length))
        self.bias = nn.Parameter(torch.Tensor(time_length, cross_length))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, add_noise=True):
        x = x.unsqueeze(3)  # [B, U, T, 1, F]
        finite = torch.isfinite(x)
        x = torch.where(finite, x, torch.zeros_like(x))

        finite_num = finite.float().sum(dim=2)  # [B, U, 1, F]
        finite_num = torch.where(
            finite_num > 0.5, finite_num, torch.ones_like(finite_num)
        )
        weighted = (x * self.weight).sum(
            dim=3, keepdim=False
        ) / finite_num  # [B, U, T, F]
        x_imputed = weighted + self.bias

        if self.training and add_noise:
            x_imputed += self.noise * torch.randn_like(x_imputed)

        return x_imputed, self.calculate_loss(x.squeeze(3), x_imputed)

    def calculate_loss(self, x, x_imputed):
        error = torch.square(x - x_imputed)

        return tensor_nanmean(error, None, False)


class Net(nn.Module):
    def __init__(
        self, time_length, cross_units, ff_time_length, ff_cross_units, batch_units, dropout, layer_norm, noise,
    ):
        nn.Module.__init__(self)
        assert cross_units[-1] + ff_cross_units[-1] == batch_units[0]

        self._time_length = time_length
        self._cross_units = cross_units
        self._ff_time_length = ff_time_length
        self._ff_cross_units = ff_cross_units
        self._batch_units = batch_units

        self.imputation_layer = SampleImputation((time_length, cross_units[0]), noise)

        cross_layers = []
        cross_units[0] = 2 * cross_units[0]
        for in_features, out_features in zip(cross_units[:-1], cross_units[1:]):
            cross_layers.append(
                Attention(in_features, out_features, dropout, layer_norm, "sample")
            )
        self.cross_layers = nn.Sequential(*cross_layers)

        # ff layers
        self.ff_imputation_layer = SampleImputation((ff_time_length, ff_cross_units[0]), noise)

        ff_cross_layers = []
        ff_cross_units[0] = 2 * ff_cross_units[0]
        for in_features, out_features in zip(ff_cross_units[:-1], ff_cross_units[1:]):
            ff_cross_layers.append(
                Attention(in_features, out_features, dropout, layer_norm, "sample")
            )
        self.ff_cross_layers = nn.Sequential(*ff_cross_layers)

        batch_layers = []
        num_layers = len(batch_units) - 1
        for i, (in_features, out_features) in enumerate(
            zip(batch_units[:-1], batch_units[1:])
        ):
            if i == num_layers - 1:
                batch_layers.append(
                    Attention(in_features, out_features, dropout, False, "batch")
                )
            else:
                batch_layers.append(
                    Attention(in_features, out_features, dropout, layer_norm, "batch")
                )
        self.batch_layers = nn.Sequential(*batch_layers)

    def _forward(self, x, ffs, add_noise):
        # x: [32,256,36,67], ffs: [32,256,36,36]
        # origin layers
        x, loss_imputation = self.imputation_layer(x, add_noise)

        x_cross = tensor_z(x, dim=-1)
        x_serial = tensor_z(x, dim=-2)
        x = torch.cat([x_cross, x_serial], dim=-1)
        x = self.cross_layers(x)  # x: [32, 256, 36, 32]
        x = x.mean(dim=2)  # x: [32,256,32]

        # ADD: ff layers
        ffs, ff_loss_imputation = self.ff_imputation_layer(ffs, add_noise)

        ffs_cross = tensor_z(ffs, dim=-1)  # [32,256,36,36]
        ffs_serial = tensor_z(ffs, dim=-2)  # [32,256,36,36]
        ffs = torch.cat([ffs_cross, ffs_serial], dim=-1)  # [32,256,36,72]
        ffs = self.ff_cross_layers(ffs)  # [32,256,36,72] -> [32,256,36,32]
        ffs = ffs.mean(dim=2)  # [32,256,36,32] -> [32,256,32]

        x = torch.cat([x, ffs], dim=-1)

        # origin layers
        x = self.batch_layers(x)
        x_centered = self.batch_centered(x)
        loss_cov = self.calculate_cov_loss(x)

        return (
            x_centered.sum(dim=2),
            x_centered,
            loss_imputation + ff_loss_imputation,
            loss_cov,
        )

    def forward(self, x, ffs):
        aggregated, risks, loss_imp, _ = self._forward(x, ffs, add_noise=True)
        _, _, _, loss_cov = self._forward(x, ffs, add_noise=False)

        return aggregated, risks, loss_imp, loss_cov

    @staticmethod
    def batch_centered(x):
        mean_x = x.mean(dim=1, keepdim=True)
        return x - mean_x

    @staticmethod
    def calculate_cov_loss(x_centered):
        num_universe = x_centered.size(1)
        num_features = x_centered.size(2)
        num_corr = num_features * (num_features - 1) / 2
        adjust = num_features * num_features / num_corr
        cov_matrix = torch.matmul(torch.transpose(x_centered, 1, 2), x_centered) / num_universe
        return torch.triu(torch.square(cov_matrix), diagonal=1).mean() * adjust


class TestModel(BaseModel):
    def __init__(
        self,
        identifier,
        sub_identifier,
        model_name,
        input_names,
        target_names,
        forward_names,
        data_shape,
        device,
        save_path,
        cross_units,
        ff_cross_units,
        batch_units,
        dropout,
        layer_norm,
        noise,
        learning_rate,
        weight_decay,
        huber_loss_positive_beta,
        huber_loss_negative_beta,
        stage,
    ):
        BaseModel.__init__(
            self,
            identifier,
            sub_identifier,
            model_name,
            input_names,
            target_names,
            forward_names,
            data_shape,
            device,
            save_path,
        )
        assert huber_loss_positive_beta > 0
        assert huber_loss_negative_beta > 0

        self.data_shape = data_shape

        time_length = self.data_shape["x"][0]
        num_cross_features = self.data_shape["x"][1]
        cross_units = [num_cross_features] + [e for e in cross_units]

        ff_time_length = self.data_shape["ffs"][0]
        num_ff_cross_features = self.data_shape["ffs"][1]
        ff_cross_units = [num_ff_cross_features] + [e for e in ff_cross_units]

        batch_units = [cross_units[-1] + ff_cross_units[-1]] + [e for e in batch_units]

        self.net = Net(
            time_length, cross_units, ff_time_length, ff_cross_units, batch_units, dropout, layer_norm, noise
        )
        self.net = self.net.to(self.device)

        self.huber_loss_positive_beta = huber_loss_positive_beta
        self.huber_loss_negative_beta = huber_loss_negative_beta

        self.optimizer = optim.Adam(
            list(self.net.parameters()), learning_rate, weight_decay=weight_decay
        )
        self.epoch = 0
        assert len(stage) == 2
        self.imp_stage = stage[0]
        self.cov_stage = stage[1]

    def optimize(self, loss: torch.Tensor) -> NoReturn:
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize_at_mini_batch(self, loss, params: list):
        self.optimize(loss)

    def optimize_at_epoch(self, params: list):
        self.epoch += 1

    def asymmetric_huber_loss(self, predict, y):
        denom = predict.numel()
        positive_part = predict >= y
        negative_part = predict < y

        positive_error = predict[positive_part] - y[positive_part]
        negative_error = predict[negative_part] - y[negative_part]

        positive_inbound = positive_error <= self.huber_loss_positive_beta
        negative_inbound = negative_error >= -self.huber_loss_negative_beta

        positive_inbound_error = positive_error[positive_inbound]
        positive_outbound_error = positive_error[~positive_inbound]
        negative_inbound_error = negative_error[negative_inbound]
        negative_outbound_error = negative_error[~negative_inbound]

        asymmetric_huber_loss = (
            torch.square(positive_inbound_error).sum() / 2
            + (
                self.huber_loss_positive_beta * positive_outbound_error
                - self.huber_loss_positive_beta ** 2 / 2
            ).sum()
            + torch.square(negative_inbound_error).sum() / 2
            + (
                -self.huber_loss_negative_beta * negative_outbound_error
                - self.huber_loss_negative_beta ** 2 / 2
            ).sum()
        ) / denom

        return asymmetric_huber_loss

    def calculate_loss(self, data: dict, is_infer=False):
        x = data["x"]
        ffs = data["ffs"]
        y = data["y"]

        x = x.to(self.device).float()
        ffs = ffs.to(self.device).float()

        num_batch, num_universe = y.size(0), y.size(1)
        y = y.view(num_batch, num_universe).to(self.device).float()
        y = tensor_z(y, dim=1)

        aggregated, risks, loss_imp, loss_cov = self.net(x, ffs)
        loss_predict = self.asymmetric_huber_loss(aggregated, y)
        if self.epoch < self.imp_stage:
            loss = loss_imp
        elif self.epoch < self.cov_stage:
            loss = loss_imp + loss_cov
        else:
            loss = loss_predict + loss_imp + loss_cov

        return loss, loss_predict, loss_imp, loss_cov

    def calculate_step_info_with_loss(self, data: dict, is_infer=False) -> dict:
        if not isinstance(data, dict):
            raise TypeError("The types of x must be dictionary.")

        loss, loss_predict, loss_imp, loss_cov = self.calculate_loss(data, is_infer)
        step_info = {
            "loss": loss - self.epoch,
            "true_loss": loss_predict + loss_imp + loss_cov,
            "loss_predict": loss_predict,
            "loss_imp": loss_imp,
            "loss_cov": loss_cov,
        }
        return step_info

    def predicts(
        self, data: dict
    ) -> Union[Tuple[str, np.ndarray], Tuple[Tuple[str, np.ndarray], ...]]:
        if not isinstance(data, dict):
            raise TypeError("The type of y should be dictionary.")

        self.set_validation_mode()
        x = data["x"]
        ffs = data["ffs"]
        x = x.to(self.device).float()
        ffs = ffs.to(self.device).float()
        _, risks, _, _ = self.net(x, ffs)

        num_features = risks.size(2)
        predicts = []

        for i in range(num_features):
            predicts.append((f"feature{i:03}", risks[0, :, i].detach().cpu().numpy()))

        return predicts


class Net2(nn.Module):
    def __init__(
        self, time_length, cross_units, serial_units, batch_units, dropout, layer_norm,
    ):
        nn.Module.__init__(self)
        assert cross_units[-1] + serial_units[-1] == batch_units[0]

        self._time_length = time_length
        self._cross_units = cross_units
        self._batch_units = batch_units

        self.imputation_layer = SampleImputation(
            (time_length, cross_units[0]), noise=0.01
        )

        cross_layers = []
        cross_units = [e for e in cross_units]
        cross_units[0] = 2 * cross_units[0]
        for in_features, out_features in zip(cross_units[:-1], cross_units[1:]):
            cross_layers.append(
                Attention(in_features, out_features, dropout, layer_norm, "sample")
            )
        self.cross_layers = nn.Sequential(*cross_layers)

        serial_layers = []
        serial_units = [e for e in serial_units]
        for in_features, out_features in zip(serial_units[:-1], serial_units[1:]):
            serial_layers.append(
                Attention(in_features, out_features, dropout, layer_norm, "sample")
            )
        self.serial_layers = nn.Sequential(*serial_layers)

        batch_layers = []
        num_layers = len(batch_units) - 1
        for i, (in_features, out_features) in enumerate(
            zip(batch_units[:-1], batch_units[1:])
        ):
            if i == num_layers - 1:
                batch_layers.append(
                    Attention(in_features, out_features, dropout, False, "batch")
                )
            else:
                batch_layers.append(
                    Attention(in_features, out_features, dropout, layer_norm, "batch")
                )
        self.batch_layers = nn.Sequential(*batch_layers)

    def forward(self, x):
        x, loss_imputation = self.imputation_layer(x)

        x_cross = tensor_z(x, dim=-1)
        x_serial = tensor_z(x, dim=-2)

        x = torch.cat([x_cross, x_serial], dim=-1)
        x_t = torch.transpose(x, 2, 3)

        x = self.cross_layers(x)
        x_t = self.serial_layers(x_t)
        x = torch.cat([x.mean(dim=2), x_t.mean(dim=2)], dim=-1)

        x = self.batch_layers(x)
        x_centered = self.batch_centered(x)
        loss_cov = self.calculate_cov_loss(x)

        return x_centered.sum(dim=2), x_centered, loss_imputation, loss_cov

    @staticmethod
    def batch_centered(x):
        mean_x = x.mean(dim=1, keepdim=True)
        return x - mean_x

    @staticmethod
    def calculate_cov_loss(x_centered):
        num_universe = x_centered.size(1)
        num_features = x_centered.size(2)
        num_corr = num_features * (num_features - 1) / 2
        adjust = num_features * num_features / num_corr
        z_centered = (
            x_centered - x_centered.mean(dim=1, keepdim=True)
        ) / x_centered.std(dim=1, keepdim=True)
        corr_matrix = (
            torch.matmul(torch.transpose(z_centered, 1, 2), z_centered) / num_universe
        )
        return torch.triu(torch.square(corr_matrix), diagonal=1).mean() * adjust


class TestModel2(BaseModel):
    def __init__(
        self,
        identifier,
        sub_identifier,
        model_name,
        input_names,
        target_names,
        forward_names,
        input_shapes,
        target_shapes,
        adversarial_info,
        device,
        save_path,
        cross_units,
        serial_units,
        batch_units,
        dropout,
        layer_norm,
        learning_rate,
        weight_decay,
        huber_loss_positive_beta,
        huber_loss_negative_beta,
    ):
        BaseModel.__init__(
            self,
            identifier,
            sub_identifier,
            model_name,
            input_names,
            target_names,
            forward_names,
            input_shapes,
            target_shapes,
            adversarial_info,
            device,
            save_path,
        )
        assert huber_loss_positive_beta > 0
        assert huber_loss_negative_beta > 0

        time_length = self.input_shapes[0]
        num_cross_features = self.input_shapes[1]
        cross_units = [num_cross_features] + [e for e in cross_units]
        serial_units = [time_length] + [e for e in serial_units]
        batch_units = [cross_units[-1] + serial_units[-1]] + [e for e in batch_units]

        self.net = Net2(
            time_length, cross_units, serial_units, batch_units, dropout, layer_norm
        )
        self.net = self.net.to(self.device)
        self.loss_cf = nn.Parameter(torch.zeros(3))

        self.huber_loss_positive_beta = huber_loss_positive_beta
        self.huber_loss_negative_beta = huber_loss_negative_beta

        self.optimizer = optim.Adam(
            list(self.net.parameters()) + [self.loss_cf],
            learning_rate,
            weight_decay=weight_decay,
        )

    def optimize(self, loss: torch.Tensor) -> NoReturn:
        loss.backward()
        self.loss_cf.grad *= -1.0
        self.optimizer.step()
        self.optimizer.zero_grad()

    def optimize_at_mini_batch(self, loss, params: list):
        self.optimize(loss)

    def optimize_at_epoch(self, params: list):
        pass

    def asymmetric_huber_loss(self, predict, y):
        denom = predict.numel()
        positive_part = predict >= y
        negative_part = predict < y

        positive_error = predict[positive_part] - y[positive_part]
        negative_error = predict[negative_part] - y[negative_part]

        positive_inbound = positive_error <= self.huber_loss_positive_beta
        negative_inbound = negative_error >= -self.huber_loss_negative_beta

        positive_inbound_error = positive_error[positive_inbound]
        positive_outbound_error = positive_error[~positive_inbound]
        negative_inbound_error = negative_error[negative_inbound]
        negative_outbound_error = negative_error[~negative_inbound]

        asymmetric_huber_loss = (
            torch.square(positive_inbound_error).sum() / 2
            + (
                self.huber_loss_positive_beta * positive_outbound_error
                - self.huber_loss_positive_beta ** 2 / 2
            ).sum()
            + torch.square(negative_inbound_error).sum() / 2
            + (
                -self.huber_loss_negative_beta * negative_outbound_error
                - self.huber_loss_negative_beta ** 2 / 2
            ).sum()
        ) / denom

        return asymmetric_huber_loss

    def calculate_loss(self, data: dict, is_infer=False):
        x = data["x"]
        y = data["y"]

        x = x.to(self.device).float()

        num_batch, num_universe = y.size(0), y.size(1)
        y = y.view(num_batch, num_universe).to(self.device).float()
        y = tensor_z(y, dim=1)

        aggregated, risks, loss_imp, loss_cov = self.net(x)
        loss_predict = self.asymmetric_huber_loss(aggregated, y)
        loss_cf = F.softmax(self.loss_cf, dim=0)
        loss = loss_cf[0] * loss_predict + loss_cf[1] * loss_imp + loss_cf[2] * loss_cov

        return loss, loss_predict, loss_imp, loss_cov, loss_cf

    def calculate_step_info_with_loss(self, data: dict, is_infer=False) -> dict:
        if not isinstance(data, dict):
            raise TypeError("The types of x must be dictionary.")

        loss, loss_predict, loss_imp, loss_cov, loss_cf = self.calculate_loss(
            data, is_infer
        )
        step_info = {
            "loss": loss,
            "loss_predict": loss_predict,
            "loss_imp": loss_imp,
            "loss_cov": loss_cov,
            "cf_predict": loss_cf[0],
            "cf_imp": loss_cf[1],
            "cf_cov": loss_cf[2],
        }
        return step_info

    def predicts(
        self, data: dict
    ) -> Union[Tuple[str, np.ndarray], Tuple[Tuple[str, np.ndarray], ...]]:
        if not isinstance(data, dict):
            raise TypeError("The type of y should be dictionary.")

        self.set_validation_mode()
        x = data["x"]
        x = x.to(self.device).float()
        _, risks, _, _ = self.net(x)

        num_features = risks.size(2)
        predicts = []

        for i in range(num_features):
            predicts.append((f"feature{i:03}", risks[0, :, i].detach().cpu().numpy()))

        return predicts
