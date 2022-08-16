from strategy_integration.components.models.deep_models.base_model import BaseModel
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, NoReturn, Iterable


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
        self.sample_shape = sample_shape # TxF : (36, 67)
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
        self,
        input_names: list,
        time_units: dict,
        cross_units: dict,
        batch_units: list,
        dropout,
        layer_norm,
        noise,
    ):
        assert set(time_units.keys()) == set(cross_units.keys()) == set(input_names)

        nn.Module.__init__(self)

        self._input_names = input_names
        self._time_units = time_units
        self._cross_units = cross_units
        self._batch_units = batch_units

        self.imputation_layers = nn.ModuleDict({
            e: SampleImputation((time_units[e], cross_units[e][0]), noise) for e in input_names
        })

        imputed_cross_units = cross_units.copy()
        for e, li in imputed_cross_units.items():
            li[0] = 2*li[0]
            imputed_cross_units[e] = li

        self.cross_layers = nn.ModuleDict({
            e: nn.Sequential(
                *[
                    Attention(in_features, out_features, dropout, layer_norm, "sample")
                    for in_features, out_features in zip(imputed_cross_units[e][:-1], imputed_cross_units[e][1:])
                ]
            ) for e in input_names
        })

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

    def _forward(self, input_data, add_noise):
        forward_data = {}
        imputation_loss = {}

        for k, x in input_data.items():
            im_data, im_loss = self.imputation_layers[k](x, add_noise)
            forward_data[k] = im_data
            imputation_loss[k] = im_loss

        for k, x in forward_data.items():
            x_cross = tensor_z(x, dim=-1)
            x_serial = tensor_z(x, dim=-2)
            x = torch.cat([x_cross, x_serial], dim=-1)
            x = self.cross_layers[k](x)
            x = x.mean(dim=2)
            forward_data[k] = x

        cat_x = torch.cat([forward_data[k] for k in self._input_names], dim=-1)
        cat_x = self.batch_layers(cat_x)
        centered_cat_x = self.batch_centered(cat_x)
        covariance_loss = self.calculate_cov_loss(centered_cat_x)
        imputation_loss = sum(imputation_loss.values())
        
        return (
            centered_cat_x.sum(dim=2),
            centered_cat_x,
            imputation_loss,
            covariance_loss,
        )

    def forward(self, input_data):
        aggregated, risks, loss_imp, _ = self._forward(input_data, add_noise=True)
        _, _, _, loss_cov = self._forward(input_data, add_noise=False)

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
        cross_units: dict,
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
        assert all((e in cross_units) for e in input_names)
        assert all((e in input_names) for e in cross_units)
        assert all(isinstance(e, list) for e in cross_units.values())

        self.data_shape = data_shape

        time_units = {e: self.data_shape[e][0] for e in input_names}
        first_cross_units = {e: self.data_shape[e][1] for e in input_names}

        first_batch_units = 0
        updated_cross_units = cross_units.copy()
        for e, li in updated_cross_units.items():
            first_batch_units += li[-1]
            first_units = first_cross_units[e]
            updated_cross_units[e] = [first_units] + li

        updated_batch_units = [first_batch_units] + [e for e in batch_units]  # [40, 40, 40, 16]

        self.net = Net(
            input_names,
            time_units, updated_cross_units, updated_batch_units,
            dropout, layer_norm, noise
        ).to(self.device)

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
        y = data["y"]
        num_batch, num_universe = y.size(0), y.size(1)
        y = y.view(num_batch, num_universe).to(self.device).float()
        y = tensor_z(y, dim=1)

        input_data = {}
        for k in self.input_names:
            v = data[k]
            input_data[k] = v.to(self.device).float()

        aggregated, risks, loss_imp, loss_cov = self.net(input_data)
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
    ) -> Union[Tuple[str, np.ndarray], Iterable[Tuple[str, np.ndarray]]]:
        if not isinstance(data, dict):
            raise TypeError("The type of y should be dictionary.")

        self.set_validation_mode()

        input_data = {}
        for k in self.input_names:
            v = data[k]
            input_data[k] = v.to(self.device).float()
        _, risks, _, _ = self.net(input_data)

        num_features = risks.size(2)
        predicts = []

        for i in range(num_features):
            predicts.append((f"feature{i:03}", risks[0, :, i].detach().cpu().numpy()))

        return predicts
