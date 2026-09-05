# pnnx model stat
# model inputshape = [1,64,3072]f32,[1,64,128]f32,[1,64,128]f32,[64,64]f32
# FLOPS = 14.974G
# memory OPS = 127.576M

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.rmsn_0 = nn.RMSNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(3072,))
        self.block_self_attn_q_proj = nn.Linear(bias=False, in_features=3072, out_features=4096)
        self.block_self_attn_k_proj = nn.Linear(bias=False, in_features=3072, out_features=1024)
        self.block_self_attn_v_proj = nn.Linear(bias=False, in_features=3072, out_features=1024)
        self.block_self_attn_o_proj = nn.Linear(bias=False, in_features=4096, out_features=3072)
        self.rmsn_1 = nn.RMSNorm(elementwise_affine=True, eps=1.000000e-5, normalized_shape=(3072,))
        self.block_mlp_gate_proj = nn.Linear(bias=False, in_features=3072, out_features=9216)
        self.block_mlp_up_proj = nn.Linear(bias=False, in_features=3072, out_features=9216)
        self.block_mlp_down_proj = nn.Linear(bias=False, in_features=9216, out_features=3072)

        archive = zipfile.ZipFile('text.pnnx.bin', 'r')
        self.rmsn_0.weight = self.load_pnnx_bin_as_parameter(archive, 'rmsn_0.weight', (3072), 'float32')
        self.block_self_attn_q_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.self_attn.q_proj.weight', (4096,3072), 'float32')
        self.block_self_attn_k_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.self_attn.k_proj.weight', (1024,3072), 'float32')
        self.block_self_attn_v_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.self_attn.v_proj.weight', (1024,3072), 'float32')
        self.block_self_attn_o_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.self_attn.o_proj.weight', (3072,4096), 'float32')
        self.rmsn_1.weight = self.load_pnnx_bin_as_parameter(archive, 'rmsn_1.weight', (3072), 'float32')
        self.block_mlp_gate_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.mlp.gate_proj.weight', (9216,3072), 'float32')
        self.block_mlp_up_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.mlp.up_proj.weight', (9216,3072), 'float32')
        self.block_mlp_down_proj.weight = self.load_pnnx_bin_as_parameter(archive, 'block.mlp.down_proj.weight', (3072,9216), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0, v_1, v_2, v_3):
        v_4 = self.rmsn_0(v_0)
        v_5 = self.block_self_attn_q_proj(v_4)
        v_6 = v_5.reshape(1, 64, 32, 128)
        v_7 = torch.transpose(v_6, dim0=1, dim1=2)
        v_8 = self.block_self_attn_k_proj(v_4)
        v_9 = v_8.reshape(1, 64, 8, 128)
        v_10 = torch.transpose(v_9, dim0=1, dim1=2)
        v_11 = self.block_self_attn_v_proj(v_4)
        v_12 = v_11.reshape(1, 64, 8, 128)
        v_13 = torch.transpose(v_12, dim0=1, dim1=2)
        v_14, v_15 = torch.tensor_split(v_7, dim=3, indices=(64,))
        v_16 = v_1.unsqueeze(1)
        v_17 = v_2.unsqueeze(1)
        v_18, v_19 = torch.tensor_split(v_16, dim=3, indices=(64,))
        v_20, v_21 = torch.tensor_split(v_17, dim=3, indices=(64,))
        v_22 = ((v_14 * v_18) - (v_15 * v_20))
        v_23 = ((v_15 * v_19) + (v_14 * v_21))
        v_24 = torch.cat((v_22, v_23), dim=-1)
        v_25, v_26 = torch.tensor_split(v_10, dim=3, indices=(64,))
        v_27 = ((v_25 * v_18) - (v_26 * v_20))
        v_28 = ((v_26 * v_19) + (v_25 * v_21))
        v_29 = torch.cat((v_27, v_28), dim=-1)
        v_30 = F.scaled_dot_product_attention(query=v_24, key=v_29, value=v_13, attn_mask=v_3, dropout_p=0.0, enable_gqa=True, is_causal=False)
        v_31 = torch.transpose(v_30, dim0=1, dim1=2)
        v_32 = v_31.reshape(1, 64, 4096)
        v_33 = self.block_self_attn_o_proj(v_32)
        v_34 = (v_0 + v_33)
        v_35 = self.rmsn_1(v_34)
        v_36 = self.block_mlp_gate_proj(v_35)
        v_37 = F.silu(v_36)
        v_38 = self.block_mlp_up_proj(v_35)
        v_39 = (v_37 * v_38)
        v_40 = self.block_mlp_down_proj(v_39)
        v_41 = (v_34 + v_40)
        return v_41

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 64, 3072, dtype=torch.float)
    v_1 = torch.rand(1, 64, 128, dtype=torch.float)
    v_2 = torch.rand(1, 64, 128, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    mod = torch.jit.trace(net, (v_0, v_1, v_2, v_3))
    mod.save("text_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 64, 3072, dtype=torch.float)
    v_1 = torch.rand(1, 64, 128, dtype=torch.float)
    v_2 = torch.rand(1, 64, 128, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    torch.onnx.export(net, (v_0, v_1, v_2, v_3), "text_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0', 'in1', 'in2', 'in3'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 64, 3072, dtype=torch.float)
    v_1 = torch.rand(1, 64, 128, dtype=torch.float)
    v_2 = torch.rand(1, 64, 128, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    import pnnx
    pnnx.export(net, "text_pnnx.py.pt", (v_0, v_1, v_2, v_3))

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 64, 3072, dtype=torch.float)
    v_1 = torch.rand(1, 64, 128, dtype=torch.float)
    v_2 = torch.rand(1, 64, 128, dtype=torch.float)
    v_3 = torch.rand(64, 64, dtype=torch.float)

    return net(v_0, v_1, v_2, v_3)

if __name__ == "__main__":
    print(test_inference())
