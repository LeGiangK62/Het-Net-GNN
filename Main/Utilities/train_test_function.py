import torch
import numpy as np


def loss_function(power_out, edge_dict, noise_matrix, size, p_cir, is_train=True, eps=1e-3, reg=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_ue, num_ap, batch_size = size
    power_out = torch.reshape(power_out, (batch_size, num_ue, -1))
    channel_matrix = edge_dict['ap', 'downlink', 'ue'][:, 0]
    power_max = power_out[:, :, 0]
    power = power_out[:, :, -1] * power_max

    # AP selection processing
    out_reshaped = edge_dict['ue', 'uplink', 'ap'][:, -1].view(-1, num_ap)
    log_val = torch.log10(1+out_reshaped/eps)
    log_val_batch = log_val.view(-1, 3, 10).permute(0,2,1)
    norm_log_val_batch = log_val_batch/np.log10(1 + 1/eps)
    sum_log = torch.sum(norm_log_val_batch, dim=2) - 1
    reg_val_batch = torch.abs(torch.sum(sum_log, dim=1))
    reg_val_mean = torch.mean(reg_val_batch)

    max_indices = torch.argmax(out_reshaped, dim=1)
    mask = torch.zeros_like(out_reshaped)
    mask.scatter_(1, max_indices.unsqueeze(1), 1)

    P = mask.view(-1, num_ue, num_ap).permute(0,2,1)

    G = torch.reshape(channel_matrix, (-1, num_ap, num_ue))
    power = power.unsqueeze(1)
    sum_rate, rate = sum_rate_calculation(P * power, P, G, noise_matrix)
    power_all = torch.sum(power, 1).unsqueeze(-1)
    sum_rate_batch = torch.sum(rate, dim=1)
    sum_power_batch = torch.sum(power_all + p_cir, dim=1)
    ee_batch = torch.div(sum_rate_batch, sum_power_batch)
    ee_mean = torch.mean(ee_batch)
    if is_train:
        return sum_rate, torch.neg(ee_mean) + reg*reg_val_mean, torch.mean(sum_power_batch), torch.neg(ee_mean)
    else:
        return torch.neg(ee_mean)


def sum_rate_calculation(power_matrix, ap_selection, channel_matrix,  noise_matrix):
    P = power_matrix
    G = channel_matrix
    desired_signal = torch.sum(torch.mul(P, G), dim=1).unsqueeze(-1)
    P_trans = P.permute(0,2,1)
    P_UE = torch.sum(P_trans, dim=2).unsqueeze(-1)  # P_UE[n] = The power n-th UE transmits
    all_received_signal = torch.matmul(G, P_UE)
    all_signal = torch.matmul(ap_selection.permute(0,2,1), all_received_signal)
    interference = -desired_signal + all_signal + noise_matrix
    rate = torch.log2(1 + torch.div(desired_signal, interference))
    sum_rate = torch.mean(torch.sum(rate, 1))
    return sum_rate, rate


def train(model, optimizer, data_loader, noise, p_cir, eps=1e-3, reg=0.01):
    model.train()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = sumRate = sumPower = ee = 0
    for batch in data_loader:
        optimizer.zero_grad()
        batch = batch.to(device_type)
        #
        num_ues = batch['ue'].num_nodes
        num_aps = batch['ap'].num_nodes
        num_edges = batch['ue', 'ap'].num_edges
        batch_size = int(num_ues * num_aps / num_edges)
        num_ues = int(num_ues / batch_size)
        num_aps = int(num_aps / batch_size)
        #
        out, edge = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_sumRate, tmp_loss, tmp_sumPower, tmp_ee = loss_function(out, edge, noise, (num_ues, num_aps, batch_size), p_cir, True, eps, reg)
        tmp_loss.backward()
        optimizer.step()
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
        sumRate += float(tmp_sumRate) * batch_size
        sumPower += float(tmp_sumPower) * batch_size
        ee += float(tmp_ee) * batch_size

    return sumRate / total_examples, total_loss / total_examples, sumPower / total_examples, ee/total_examples


def test(model, data_loader, noise, p_cir):
    model.eval()
    device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_examples = total_loss = 0
    for batch in data_loader:
        batch = batch.to(device_type)
        #
        num_ues = batch['ue'].num_nodes
        num_aps = batch['ap'].num_nodes
        num_edges = batch['ue', 'ap'].num_edges
        batch_size = int(num_ues * num_aps / num_edges)
        num_ues = int(num_ues / batch_size)
        num_aps = int(num_aps / batch_size)
        #
        out, edge = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
        out = out['ue']
        tmp_loss = loss_function(out, edge, noise, (num_ues, num_aps, batch_size), p_cir, False)
        total_examples += batch_size
        total_loss += float(tmp_loss) * batch_size
    return total_loss / total_examples
