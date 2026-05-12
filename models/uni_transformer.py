import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, knn_graph, global_mean_pool
from torch_geometric.nn import SAGPooling, GCNConv
from torch_scatter import scatter_softmax, scatter_sum, scatter_mean

from models.common import GaussianSmearing, MLP, batch_hybrid_edge_connection, outer_product, compose_context


class BaseX2HAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r', out_fc=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.act_fn = act_fn
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.ew_net_type = ew_net_type
        self.out_fc = out_fc

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim
        self.hk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.hv_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)

        self.hq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())
        elif ew_net_type == 'm':
            self.ew_net = nn.Sequential(nn.Linear(output_dim, 1), nn.Sigmoid())

        if self.out_fc:
            self.node_output = MLP(2 * hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.hk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.hv_func(kv_input)

        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = self.ew_net(v[..., :self.hidden_dim])
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w
        v = v.view(-1, self.n_heads, self.output_dim // self.n_heads)

        q = self.hq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0,
                                dim_size=N)

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)
        output = output.view(-1, self.output_dim)
        if self.out_fc:
            output = self.node_output(torch.cat([output, h], -1))

        output = output + h
        return output


class BaseH2XAttLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, edge_feat_dim, r_feat_dim,
                 act_fn='relu', norm=True, ew_net_type='r'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.r_feat_dim = r_feat_dim
        self.act_fn = act_fn
        self.ew_net_type = ew_net_type

        kv_input_dim = input_dim * 2 + edge_feat_dim + r_feat_dim

        self.xk_func = MLP(kv_input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(kv_input_dim, self.n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(input_dim, output_dim, hidden_dim, norm=norm, act_fn=act_fn)
        if ew_net_type == 'r':
            self.ew_net = nn.Sequential(nn.Linear(r_feat_dim, 1), nn.Sigmoid())

    def forward(self, h, rel_x, r_feat, edge_feat, edge_index, e_w=None):
        N = h.size(0)
        src, dst = edge_index
        hi, hj = h[dst], h[src]

        kv_input = torch.cat([r_feat, hi, hj], -1)
        if edge_feat is not None:
            kv_input = torch.cat([edge_feat, kv_input], -1)

        k = self.xk_func(kv_input).view(-1, self.n_heads, self.output_dim // self.n_heads)
        v = self.xv_func(kv_input)
        if self.ew_net_type == 'r':
            e_w = self.ew_net(r_feat)
        elif self.ew_net_type == 'm':
            e_w = 1.
        elif e_w is not None:
            e_w = e_w.view(-1, 1)
        else:
            e_w = 1.
        v = v * e_w

        v = v.unsqueeze(-1) * rel_x.unsqueeze(1)  # (xi - xj) [n_edges, n_heads, 3]
        q = self.xq_func(h).view(-1, self.n_heads, self.output_dim // self.n_heads)

        alpha = scatter_softmax((q[dst] * k / np.sqrt(k.shape[-1])).sum(-1), dst, dim=0, dim_size=N)  # (E, heads)

        m = alpha.unsqueeze(-1) * v
        output = scatter_sum(m, dst, dim=0, dim_size=N)
        return output.mean(1)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn='relu', norm=True,
                 num_x2h=1, num_h2x=1, r_min=0., r_max=10., num_node_types=8,
                 ew_net_type='r', x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.edge_feat_dim = edge_feat_dim
        self.num_r_gaussian = num_r_gaussian
        self.norm = norm
        self.act_fn = act_fn
        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.r_min, self.r_max = r_min, r_max
        self.num_node_types = num_node_types
        self.ew_net_type = ew_net_type
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup

        self.distance_expansion = GaussianSmearing(self.r_min, self.r_max, num_gaussians=num_r_gaussian)

        self.x2h_layers = nn.ModuleList()
        for i in range(self.num_x2h):
            self.x2h_layers.append(
                BaseX2HAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 5,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type, out_fc=self.x2h_out_fc)
            )
        self.h2x_layers = nn.ModuleList()
        for i in range(self.num_h2x):
            self.h2x_layers.append(
                BaseH2XAttLayer(hidden_dim, hidden_dim, hidden_dim, n_heads, edge_feat_dim,
                                r_feat_dim=num_r_gaussian * 5,
                                act_fn=act_fn, norm=norm,
                                ew_net_type=self.ew_net_type)
            )

    def forward(self, h, x, edge_attr, edge_index, mask_ligand, e_w=None, fix_x=False):
        src, dst = edge_index
        if self.edge_feat_dim > 0:
            edge_feat = edge_attr
        else:
            edge_feat = None

        rel_x = x[dst] - x[src]
        dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        h_in = h
        for i in range(self.num_x2h):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            h_out = self.x2h_layers[i](h_in, dist_feat, edge_feat, edge_index, e_w=e_w)
            h_in = h_out
        x2h_out = h_in

        new_h = h if self.sync_twoup else x2h_out
        for i in range(self.num_h2x):
            dist_feat = self.distance_expansion(dist)
            dist_feat = outer_product(edge_attr, dist_feat)
            delta_x = self.h2x_layers[i](new_h, rel_x, dist_feat, edge_feat, edge_index, e_w=e_w)
            if not fix_x:
                # x = x + delta_x * mask_ligand[:, None]  # only ligand positions will be updated
                x = x + delta_x # update both pocket and ligand positions
            rel_x = x[dst] - x[src]
            dist = torch.norm(rel_x, p=2, dim=-1, keepdim=True)

        return x2h_out, x


class GVPLayer(nn.Module):
    """
    纯 PyTorch 实现的 Geometric Vector Perceptron
    输入  : (s, V)   s:[B, n_s] , V:[B, n_v, 3]
    输出  : (s', V') s':[B, n_s_out], V':[B, n_v_out, 3]
    """
    def __init__(self,
                 in_dims,      # (n_s, n_v)
                 out_dims,     # (n_s_out, n_v_out)
                 h_dim=None,   # 中间向量通道数, 默认 max(n_v, n_v_out)
                 scalar_act=F.relu,
                 vector_act=torch.sigmoid,
                 vector_gate=False):
        super().__init__()
        n_s, n_v     = in_dims
        n_s_out, n_v_out = out_dims
        self.vector_gate = vector_gate

        # ----- 向量支路 -----
        if n_v > 0:
            h_dim = h_dim or max(n_v, n_v_out)
            self.W_h = nn.Linear(n_v, h_dim, bias=False)   # V -> V_h
            if n_v_out  > 0:
                self.W_v = nn.Linear(h_dim, n_v_out, bias=False)  # V_h -> V'
            # ----- 标量支路 -----
            self.W_s = nn.Linear(h_dim + n_s, n_s_out)
            if vector_gate and n_v_out > 0:
                self.W_sv = nn.Linear(n_s_out, n_v_out)
        else:  # 没有向量输入
            self.W_s = nn.Linear(n_s, n_s_out)

        self.scalar_act = scalar_act
        self.vector_act = vector_act
        self.n_v_out = n_v_out

    def forward(self, data):
        """
        data : (s, V) 或 者仅 s（当 n_v==0）
        """
        # 情形一：存在向量输入
        if isinstance(data, tuple):
            s, V = data                           # V:[B, n_v, 3]
            Vt   = V.transpose(-1, -2)            # -> [B, 3, n_v]
            V_h  = self.W_h(Vt)                   # [B, 3, h]
            # 取行向量范数 ‖V_h‖₂   [B, h]
            Vh_norm = torch.norm(V_h, dim=-2)     
            # --- 更新标量 ---
            s_cat   = torch.cat([s, Vh_norm], dim=-1)
            s_out   = self.W_s(s_cat)
            # --- 更新向量 ---
            if self.n_v_out > 0:
                V_prime = self.W_v(V_h)           # [B, 3, n_v_out]
                V_prime = V_prime.transpose(-1, -2)  # [B, n_v_out, 3]

                if self.vector_gate:
                    gate = self.W_sv(self.vector_act(s_out))
                    V_prime = V_prime * torch.sigmoid(gate).unsqueeze(-1)
                else:
                    scale  = self.vector_act(torch.norm(
                              V_prime, dim=-1, keepdim=True))
                    V_prime = V_prime * scale
            else:
                V_prime = None
        # 情形二：纯标量
        else:
            s_out   = self.W_s(data)
            V_prime = None

        if self.scalar_act is not None:
            s_out = self.scalar_act(s_out)

        return (s_out, V_prime) if V_prime is not None else s_out


class SAGPoolNet(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, ratio=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=ratio)  # 注意力打分 + Top-k
        self.lin   = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)

        return self.lin(x)


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads=1, k=32,
                 num_r_gaussian=50, edge_feat_dim=0, num_node_types=8, act_fn='relu', norm=True,
                 cutoff_mode='radius', ew_net_type='r',
                 num_init_x2h=1, num_init_h2x=0, num_x2h=1, num_h2x=1, r_max=10., x2h_out_fc=True, sync_twoup=False):
        super().__init__()
        # Build the network
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.num_r_gaussian = num_r_gaussian
        self.edge_feat_dim = edge_feat_dim
        self.act_fn = act_fn
        self.norm = norm
        self.num_node_types = num_node_types
        self.cutoff_mode = cutoff_mode
        self.k = k
        self.ew_net_type = ew_net_type

        self.num_x2h = num_x2h
        self.num_h2x = num_h2x
        self.num_init_x2h = num_init_x2h
        self.num_init_h2x = num_init_h2x
        self.r_max = r_max
        self.x2h_out_fc = x2h_out_fc
        self.sync_twoup = sync_twoup
        self.distance_expansion = GaussianSmearing(0., r_max, num_gaussians=num_r_gaussian)
        if self.ew_net_type == 'global':
            self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)

        self.init_h_emb_layer = self._build_init_h_layer()
        self.base_block = self._build_share_blocks()
        # self.lig_gvp_layer = GVPLayer(
        #     in_dims=(self.hidden_dim, 1),
        #     out_dims=(self.hidden_dim, 1),
        #     vector_gate=True,
        # )
        self.prot_gvp_layer = GVPLayer(
            in_dims=(self.hidden_dim, 1),
            out_dims=(self.hidden_dim//2, 4),
            vector_gate=True,
        )
        self.prot_sag_layer = SAGPoolNet(
            in_dim=self.hidden_dim + 3,  # +3 for protein residue position
            hidden_dim=self.hidden_dim + 3,
            ratio=0.5,  # 50% pooling
        )

    def __repr__(self):
        return f'UniTransformerO2(num_blocks={self.num_blocks}, num_layers={self.num_layers}, n_heads={self.n_heads}, ' \
               f'act_fn={self.act_fn}, norm={self.norm}, cutoff_mode={self.cutoff_mode}, ew_net_type={self.ew_net_type}, ' \
               f'init h emb: {self.init_h_emb_layer.__repr__()} \n' \
               f'base block: {self.base_block.__repr__()} \n' \
               f'edge pred layer: {self.edge_pred_layer.__repr__() if hasattr(self, "edge_pred_layer") else "None"}) '

    def _build_init_h_layer(self):
        layer = AttentionLayerO2TwoUpdateNodeGeneral(
            self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn, norm=self.norm,
            num_x2h=self.num_init_x2h, num_h2x=self.num_init_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
            ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
        )
        return layer

    def _build_share_blocks(self):
        base_block = []
        for l_idx in range(self.num_layers):
            layer = AttentionLayerO2TwoUpdateNodeGeneral(
                self.hidden_dim, self.n_heads, self.num_r_gaussian, self.edge_feat_dim, act_fn=self.act_fn,
                norm=self.norm,
                num_x2h=self.num_x2h, num_h2x=self.num_h2x, r_max=self.r_max, num_node_types=self.num_node_types,
                ew_net_type=self.ew_net_type, x2h_out_fc=self.x2h_out_fc, sync_twoup=self.sync_twoup,
            )
            base_block.append(layer)
        return nn.ModuleList(base_block)

    def _connect_edge(self, x, mask_ligand, batch):
        if self.cutoff_mode == 'radius':
            edge_index = radius_graph(x, r=self.r, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            edge_index = knn_graph(x, k=self.k, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'hybrid':
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, mask_ligand=mask_ligand, batch=batch, add_p_index=True)
        else:
            raise ValueError(f'Not supported cutoff mode: {self.cutoff_mode}')
        return edge_index

    @staticmethod
    # def _build_edge_type(edge_index, mask_ligand):
    #     src, dst = edge_index
    #     edge_type = torch.zeros(len(src)).to(edge_index)
    #     n_src = mask_ligand[src] == 1
    #     n_dst = mask_ligand[dst] == 1
    #     edge_type[n_src & n_dst] = 0
    #     edge_type[n_src & ~n_dst] = 1
    #     edge_type[~n_src & n_dst] = 2
    #     edge_type[~n_src & ~n_dst] = 3
    #     edge_type = F.one_hot(edge_type, num_classes=4)
    #     return edge_type
    def _build_edge_type(edge_index,
                        mask_ligand: torch.Tensor,
                        atom_to_residue: torch.Tensor,
                        batch_all: torch.Tensor):
        """
        Parameters
        ----------
        edge_index   : (2, E) LongTensor
        mask_ligand  : (N,)   Bool / {0,1}; 1 ⇒ ligand atom
        atom_to_residue : (N_protein_atom,)   LongTensor giving residue index per atom (resets for each batch)
        batch_all    : (N,)   LongTensor giving batch index per atom
        Returns
        -------
        edge_type_onehot : (E, 5) FloatTensor, five categories listed above
        """
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        
        # 创建完整的残基映射数组 (包含所有原子)
        mask_protein = ~mask_ligand
        full_atom_to_residue = torch.full_like(batch_all, -1)  # 初始化为-1 (配体原子)
        full_atom_to_residue[mask_protein] = atom_to_residue   # 填入蛋白质原子的残基ID

        # 找到每个batch的残基数量偏移
        batch_residue_offsets = []
        current_offset = 0

        for batch_idx in torch.unique(batch_all):
            batch_mask = batch_all == batch_idx
            # 只考虑蛋白质原子的残基ID
            protein_mask_batch = mask_protein & batch_mask
            if protein_mask_batch.sum() > 0:
                batch_protein_residue = full_atom_to_residue[protein_mask_batch]
                num_residues_in_batch = batch_protein_residue.max().item() + 1
            else:
                num_residues_in_batch = 0
            batch_residue_offsets.append(current_offset)
            current_offset += num_residues_in_batch

        # 创建全局唯一的残基ID
        global_residue_idx = full_atom_to_residue.clone()
        for i, batch_idx in enumerate(torch.unique(batch_all)):
            batch_mask = batch_all == batch_idx
            protein_mask_batch = mask_protein & batch_mask
            if protein_mask_batch.sum() > 0:
                global_residue_idx[protein_mask_batch] += batch_residue_offsets[i]

        # convenience masks
        src_lig = mask_ligand[src] == 1
        dst_lig = mask_ligand[dst] == 1
        both_prot = mask_protein[src] & mask_protein[dst]

        # 0️⃣ ligand–ligand
        edge_type[src_lig & dst_lig] = 0
        # 1️⃣ ligand → protein
        edge_type[src_lig & ~dst_lig] = 1
        # 2️⃣ protein → ligand
        edge_type[~src_lig & dst_lig] = 2

        # split the old "protein–protein" case into two using global residue IDs
        same_res = global_residue_idx[src] == global_residue_idx[dst]
        # 3️⃣ protein–protein, same residue
        edge_type[both_prot &  same_res] = 3
        # 4️⃣ protein–protein, across residues
        edge_type[both_prot & ~same_res] = 4

        # one-hot for downstream embedding
        edge_type_onehot = F.one_hot(edge_type, num_classes=5)

        return edge_type_onehot

    @staticmethod
    def _build_prot_edge_index(prot_pos, k=16):
        """
        Fallback: build a quick kNN graph if you don't already have edge_index.
        Using torch_geometric.nn.pool.knn_graph (import inside to avoid global dep).
        """
        # pos: (N_atoms, 3)
        edge_index = knn_graph(prot_pos, k=k, loop=False)  # (2, E)

        return edge_index

    # def forward(self, h, x, mask_ligand, batch, return_all=False, fix_x=False):

    #     all_x = [x]
    #     all_h = [h]

    #     for b_idx in range(self.num_blocks):
    #         edge_index = self._connect_edge(x, mask_ligand, batch) # use knn to build edges, mask_ligand will not impact the edge_index
    #         src, dst = edge_index

    #         edge_type = self._build_edge_type(edge_index, mask_ligand)
    #         if self.ew_net_type == 'global':
    #             dist = torch.norm(x[dst] - x[src], p=2, dim=-1, keepdim=True)
    #             dist_feat = self.distance_expansion(dist)
    #             logits = self.edge_pred_layer(dist_feat)
    #             e_w = torch.sigmoid(logits)
    #         else:
    #             e_w = None

    #         for l_idx, layer in enumerate(self.base_block):
    #             h, x = layer(h, x, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
    #         all_x.append(x)
    #         all_h.append(h)

    #     outputs = {'x': x, 'h': h}
    #     if return_all:
    #         outputs.update({'all_x': all_x, 'all_h': all_h})
    #     return outputs

    def forward(self, h_protein, h_ligand, protein_pos, ligand_pos, batch_protein, batch_ligand,
                protein_atom_to_aa_group, return_all=False, fix_x=False):

        # Compose context
        h_all, pos_all, batch_all, mask_ligand, mask_protein, _ = compose_context(
            h_protein=h_protein,
            h_ligand=h_ligand,
            pos_protein=protein_pos,
            pos_ligand=ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            hbap_protein=None,
            hbap_ligand=None,
        )

        all_pos_list = [pos_all]
        all_h_list = [h_all]

        for b_idx in range(self.num_blocks):
            edge_index = self._connect_edge(pos_all, mask_ligand, batch_all) # use knn to build edges, mask_ligand will not impact the edge_index
            src, dst = edge_index

            edge_type = self._build_edge_type(
                edge_index=edge_index,
                mask_ligand=mask_ligand,
                atom_to_residue=protein_atom_to_aa_group,
                batch_all=batch_all,
            )
            if self.ew_net_type == 'global':
                dist = torch.norm(pos_all[dst] - pos_all[src], p=2, dim=-1, keepdim=True)
                dist_feat = self.distance_expansion(dist)
                logits = self.edge_pred_layer(dist_feat)
                e_w = torch.sigmoid(logits)
            else:
                e_w = None

            for l_idx, layer in enumerate(self.base_block):
                h_all, pos_all = layer(h_all, pos_all, edge_type, edge_index, mask_ligand, e_w=e_w, fix_x=fix_x)
            all_pos_list.append(pos_all)
            all_h_list.append(h_all)

        # ligand_pos, ligand_h = pos_all[mask_ligand], h_all[mask_ligand]
        # ligand_pos = ligand_pos.unsqueeze(1) if ligand_pos.dim() == 2 else ligand_pos
        # final_ligand_h, final_ligand_pos = self.lig_gvp_layer((ligand_h, ligand_pos))
        # # Reshape the ligand_pos from [N, 1, 3] to [N, 3] for concatenation
        # final_ligand_pos = final_ligand_pos.reshape(final_ligand_pos.shape[0], -1)
        final_ligand_pos, final_ligand_h = pos_all[mask_ligand], h_all[mask_ligand]

        protein_pos, protein_h = pos_all[mask_protein], h_all[mask_protein]
        # protein_pos = protein_pos.unsqueeze(1) if protein_pos.dim() == 2 else protein_pos
        # protein_h, protein_pos = self.prot_gvp_layer((protein_h, protein_pos))
        # # Reshape the protein_pos from [N, 4, 3] to [N, 12] for concatenation
        # protein_pos = protein_pos.reshape(protein_pos.shape[0], -1)
        # h_protein_update = torch.concat([protein_h, protein_pos], dim=-1)
        # h_residue_update = self._aggregate_atom_to_residue(h_protein_update, protein_atom_to_aa_group, batch_protein)
        h_protein_update = torch.concat([protein_h, protein_pos], dim=-1)
        h_residue_update = self._aggregate_atom_to_residue(
            atom_features=h_protein_update,
            atom_to_residue=protein_atom_to_aa_group,
            batch_protein=batch_protein,
            atom_pos=protein_pos,
        )

        outputs = {
            'ligand_pos': final_ligand_pos,
            'ligand_h': final_ligand_h,
            'residue_h': h_residue_update,
        }

        return outputs

    # def _aggregate_atom_to_residue(self, atom_features, atom_to_residue, batch_protein):
    #     """
    #     Aggregate atom features to residue features.
    #     考虑batch信息的原子到残基特征聚合

    #     输入:
    #     - atom_features: (N_protein, feature_dim) - 原子特征 (可能包含位置信息)
    #     - atom_to_residue: (N_protein,) - 原子到残基的映射 (每个batch内残基ID从0开始)
    #     - batch_protein: (N_protein,) - 原子的batch索引

    #     输出:
    #     - residue_features: (N_residue_total, feature_dim) - 残基特征
    #     """
    #     # 找到每个batch的残基数量偏移
    #     batch_residue_offsets = []
    #     current_offset = 0

    #     for batch_idx in torch.unique(batch_protein):
    #         batch_mask = batch_protein == batch_idx
    #         batch_atom_to_residue = atom_to_residue[batch_mask]
    #         num_residues_in_batch = batch_atom_to_residue.max().item() + 1
    #         batch_residue_offsets.append(current_offset)
    #         current_offset += num_residues_in_batch

    #     # 创建全局唯一的残基ID
    #     global_residue_idx = atom_to_residue.clone()
    #     for i, batch_idx in enumerate(torch.unique(batch_protein)):
    #         batch_mask = batch_protein == batch_idx
    #         global_residue_idx[batch_mask] += batch_residue_offsets[i]

    #     # 总残基数量
    #     total_num_residues = current_offset

    #     # 使用全局残基ID进行聚合
    #     residue_features = scatter_mean(atom_features, global_residue_idx, dim=0, dim_size=total_num_residues)

    #     return residue_features

    def _aggregate_atom_to_residue(self,
                                   atom_features: torch.Tensor,       # (N_atoms, F)
                                   atom_to_residue: torch.Tensor,     # (N_atoms,) local residue id per *protein*
                                   batch_protein: torch.Tensor,       # (N_atoms,) protein id per atom
                                   atom_pos: torch.Tensor,     # (N_atoms, 3) optional (only needed if building edges)
                                   knn_k: int = 16):
        """
        Returns
        -------
        residue_features : (N_residues_total, F_res)
        batch_residue    : (N_residues_total,) protein id for each residue
        residue_id_global: (N_atoms,) global residue index (if needed downstream)
        """
        prot_edge_index = self._build_prot_edge_index(prot_pos=atom_pos, k=knn_k)

        # 找到每个batch的残基数量偏移
        batch_residue_offsets = []
        current_offset = 0

        for batch_idx in torch.unique(batch_protein):
            batch_mask = batch_protein == batch_idx
            batch_atom_to_residue = atom_to_residue[batch_mask]
            num_residues_in_batch = batch_atom_to_residue.max().item() + 1
            batch_residue_offsets.append(current_offset)
            current_offset += num_residues_in_batch

        # 创建全局唯一的残基ID
        global_residue_idx = atom_to_residue.clone()
        for i, batch_idx in enumerate(torch.unique(batch_protein)):
            batch_mask = batch_protein == batch_idx
            global_residue_idx[batch_mask] += batch_residue_offsets[i]

        # 总残基数量
        total_num_residues = current_offset

        src, dst = prot_edge_index
        same_residue_edge_mask = (global_residue_idx[src] == global_residue_idx[dst])
        edge_index_residue_local = prot_edge_index[:, same_residue_edge_mask]

        residue_features = self.prot_sag_layer(
            x=atom_features,
            edge_index=edge_index_residue_local,
            batch=global_residue_idx,
        )

        return residue_features
