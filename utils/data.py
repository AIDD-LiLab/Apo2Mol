import os
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from typing import Dict, List, Tuple
from numpy import dot, transpose
from numpy.linalg import svd, det
from scipy.spatial.transform import Rotation
from Bio import PDB
from Bio.PDB.vectors import Vector
from scipy.spatial.transform import Rotation as R
import kornia as K
import quaternion


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}


# CHI_ORDER = ["chi1", "altchi1", "chi2", "altchi2", "chi3", "chi4", "chi5"]
# MAX_CHI   = 7
CHI_ORDER = ["chi1", "chi2", "chi3", "chi4", "chi5"]
MAX_CHI   = 5

# —— χ-角原子定义表 ——
chi_atoms = dict(
    chi1=dict(
        ARG=['N', 'CA', 'CB', 'CG'],
        ASN=['N', 'CA', 'CB', 'CG'],
        ASP=['N', 'CA', 'CB', 'CG'],
        CYS=['N', 'CA', 'CB', 'SG'],
        GLN=['N', 'CA', 'CB', 'CG'],
        GLU=['N', 'CA', 'CB', 'CG'],
        HIS=['N', 'CA', 'CB', 'CG'],
        ILE=['N', 'CA', 'CB', 'CG1'],
        LEU=['N', 'CA', 'CB', 'CG'],
        LYS=['N', 'CA', 'CB', 'CG'],
        MET=['N', 'CA', 'CB', 'CG'],
        PHE=['N', 'CA', 'CB', 'CG'],
        PRO=['N', 'CA', 'CB', 'CG'],
        SER=['N', 'CA', 'CB', 'OG'],
        THR=['N', 'CA', 'CB', 'OG1'],
        TRP=['N', 'CA', 'CB', 'CG'],
        TYR=['N', 'CA', 'CB', 'CG'],
        VAL=['N', 'CA', 'CB', 'CG1'],
    ),
    # altchi1=dict(
    #     VAL=['N', 'CA', 'CB', 'CG2'],
    # ),
    chi2=dict(
        ARG=['CA', 'CB', 'CG', 'CD'],
        ASN=['CA', 'CB', 'CG', 'OD1'],
        ASP=['CA', 'CB', 'CG', 'OD1'],
        GLN=['CA', 'CB', 'CG', 'CD'],
        GLU=['CA', 'CB', 'CG', 'CD'],
        HIS=['CA', 'CB', 'CG', 'ND1'],
        ILE=['CA', 'CB', 'CG1', 'CD1'],
        LEU=['CA', 'CB', 'CG', 'CD1'],
        LYS=['CA', 'CB', 'CG', 'CD'],
        MET=['CA', 'CB', 'CG', 'SD'],
        PHE=['CA', 'CB', 'CG', 'CD1'],
        PRO=['CA', 'CB', 'CG', 'CD'],
        TRP=['CA', 'CB', 'CG', 'CD1'],
        TYR=['CA', 'CB', 'CG', 'CD1'],
    ),
    # altchi2=dict(
    #     ASP=['CA', 'CB', 'CG', 'OD2'],
    #     LEU=['CA', 'CB', 'CG', 'CD2'],
    #     PHE=['CA', 'CB', 'CG', 'CD2'],
    #     TYR=['CA', 'CB', 'CG', 'CD2'],
    # ),
    chi3=dict(
        ARG=['CB', 'CG', 'CD', 'NE'],
        GLN=['CB', 'CG', 'CD', 'OE1'],
        GLU=['CB', 'CG', 'CD', 'OE1'],
        LYS=['CB', 'CG', 'CD', 'CE'],
        MET=['CB', 'CG', 'SD', 'CE'],
    ),
    chi4=dict(
        ARG=['CG', 'CD', 'NE', 'CZ'],
        LYS=['CG', 'CD', 'CE', 'NZ'],
    ),
    chi5=dict(
        ARG=['CD', 'NE', 'CZ', 'NH1'],
    ),
)

# ──────────────────── 几何工具 ────────────────────
def get_align_rotran(coords: np.ndarray,
                     reference_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    只接受 (3,3) 坐标：N, CA, C。
    返回:
        tran : (3,)  平移向量
        rot  : (3,3) 旋转矩阵 (row-vector 右乘约定)
    """
    av1 = coords.mean(0, keepdims=True)
    av2 = reference_coords.mean(0, keepdims=True)
    # coords          -= av1
    coords -= av2
    reference_coords-= av2

    a   = dot(transpose(coords), reference_coords)
    u, _, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    if det(rot) < 0:            # 纠正反射
        vt[2] *= -1
        rot = transpose(dot(transpose(vt), transpose(u)))

    tran = av2.squeeze() - dot(av1.squeeze(), rot)
    return tran, rot


def get_align_rotran_quat(coords: np.ndarray, reference_coords: np.ndarray):
    # 与原函数相同的 Kabsch 对齐处理
    av1 = coords.mean(0, keepdims=True)
    av2 = reference_coords.mean(0, keepdims=True)
    # coords          -= av1
    coords -= av2
    reference_coords-= av2

    a   = dot(transpose(coords), reference_coords)
    u, _, vt = svd(a)
    rot = transpose(dot(transpose(vt), transpose(u)))
    if det(rot) < 0:            # 纠正反射
        vt[2] *= -1
        rot = transpose(dot(transpose(vt), transpose(u)))

    tran = av2.squeeze() - dot(av1.squeeze(), rot)

    # 从旋转矩阵构造单位四元数并设置dtype为float32
    q = quaternion.from_rotation_matrix(rot.T)
    return tran, q


def get_align_rotran_kabsch(coords: np.ndarray,
                           reference_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用Kabsch算法计算最优旋转和平移，将coords对齐到reference_coords
    
    Args:
        coords: (N, 3) 源坐标 (需要被变换的坐标)
        reference_coords: (N, 3) 目标坐标

    Returns:
        tran: (3,) 平移向量
        rot: (3,3) 旋转矩阵

    变换关系: reference_coords = coords @ rot.T + tran
    """
    assert coords.shape == reference_coords.shape
    assert coords.shape[1] == 3
    
    # 1. 计算质心
    centroid_coords = coords.mean(axis=0)
    centroid_reference = reference_coords.mean(axis=0)
    
    # # 2. 将坐标移动到各自质心
    coords_centered = coords - centroid_coords
    reference_centered = reference_coords - centroid_reference

    # 3. 计算协方差矩阵 H = P^T * Q
    H = coords_centered.T @ reference_centered
    # H = coords.T @ reference_coords

    # 4. SVD分解
    U, S, Vt = np.linalg.svd(H)

    # 5. 计算旋转矩阵
    rot = Vt.T @ U.T

    # 6. 确保是旋转矩阵而不是反射 (det = 1)
    if np.linalg.det(rot) < 0:
        Vt[-1, :] *= -1
        rot = Vt.T @ U.T

    # 7. 计算平移向量
    tran = centroid_reference - centroid_coords @ rot.T
    # tran = reference_coords.mean(axis=0) - coords.mean(axis=0) @ rot.T
    return tran, rot

# def dihedral(p0, p1, p2, p3) -> float:
#     b0, b1, b2 = p1-p0, p2-p1, p3-p2
#     b1 /= np.linalg.norm(b1)
#     v = b0 - b1*np.dot(b0, b1)
#     w = b2 - b1*np.dot(b2, b1)
#     return np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))

# ─────────────── χ-角（单残基） ───────────────
def compute_chis_one_res(atom_coord: Dict[str, np.ndarray],
                         resname: str) -> Tuple[np.ndarray, np.ndarray]:
    chis = np.zeros(MAX_CHI, np.float32)
    mask = np.zeros(MAX_CHI, np.int8)

    for i, chi_key in enumerate(CHI_ORDER):
        atoms = chi_atoms.get(chi_key, {}).get(resname)
        if not atoms:
            continue
        if all(a in atom_coord for a in atoms):
            p = [atom_coord[a] for a in atoms]                    # 四个 np.array
            angle = PDB.calc_dihedral(*(Vector(v) for v in p))        # → rad
            chis[i] = angle % (2 * np.pi)                         # 映射到 0–2π
            mask[i] = 1

    # ---- altχ 处理与 dynamicBind 保持一致 ----
    # if resname in chi_atoms["altchi1"]:
    #     idx_main, idx_alt = 0, 1
    #     if mask[idx_main] and mask[idx_alt]:
    #         a, b = chis[idx_main], chis[idx_alt]
    #         chis[idx_main], chis[idx_alt] = max(a, b), min(a, b)
    #     else:
    #         chis[idx_main] = chis[idx_alt] = 0.0
    #         mask[idx_main] = mask[idx_alt] = 0

    # if resname in chi_atoms["altchi2"]:
    #     idx_main, idx_alt = 2, 3
    #     if mask[idx_main] and mask[idx_alt]:
    #         a, b = chis[idx_main], chis[idx_alt]
    #         chis[idx_main], chis[idx_alt] = max(a, b), min(a, b)
    #     else:
    #         chis[idx_main] = chis[idx_alt] = 0.0
    #         mask[idx_main] = mask[idx_alt] = 0

    return chis, mask

# ─────────────── 主接口：批量残基 ───────────────
def compute_residue_transforms(
    protein_pos_apo         : np.ndarray,   # (N,3)
    protein_pos_holo        : np.ndarray,   # (N,3)
    protein_atom_name       : List[str],
    protein_atom_to_aa_name : List[str],
    protein_atom_to_aa_group: np.ndarray,
) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:

    residue_index: Dict[int, List[int]] = {}
    for i, gid in enumerate(protein_atom_to_aa_group):
        residue_index.setdefault(int(gid), []).append(i)

    quats, rot_vecs, translations = [], [], []
    chi_apo_all, chi_holo_all, chi_mask_all = [], [], []

    for idxs in residue_index.values():
        # —— 非氢原子坐标
        atom_apo, atom_holo = {}, {}
        for i in idxs:
            name = protein_atom_name[i]
            if name.startswith("H"):
                continue
            atom_apo [name] = protein_pos_apo[i]
            atom_holo[name] = protein_pos_holo[i]

        # —— 只保留 N, CA, C
        try:
            P = np.stack([atom_apo [a] for a in ("N", "CA", "C")])
            Q = np.stack([atom_holo[a] for a in ("N", "CA", "C")])
        except KeyError:
            continue

        # t_, R = get_align_rotran(P.copy(), Q.copy())
        t, R = get_align_rotran(Q.copy(), P.copy())  # Q → P
        # t, R = get_align_rotran_kabsch(Q.copy(), P.copy())
        rotvec = Rotation.from_matrix(R.T).as_rotvec()
        # Convert rot_vec to quaternion
        q = quaternion.from_rotation_matrix(R.T)  # 注意：R.T 是因为 Korn
        q_array = np.array([q.w, q.x, q.y, q.z])  # 四元数转换为数组

        resname           = protein_atom_to_aa_name[idxs[0]]
        chi_apo, mask     = compute_chis_one_res(atom_apo,  resname)
        chi_holo, _       = compute_chis_one_res(atom_holo, resname)

        translations.append(t)
        quats.append(q_array)
        rot_vecs.append(rotvec)
        chi_apo_all.append(chi_apo)
        chi_holo_all.append(chi_holo)
        chi_mask_all.append(mask)

    # rotations    = torch.tensor(rot_vecs, dtype=torch.float32)  # (M,3)
    rotations    = torch.tensor(quats, dtype=torch.float32)  # (M,4) 四元数
    rot_vecs    = torch.tensor(rot_vecs, dtype=torch.float32)  # (M,3)
    translations = torch.tensor(translations, dtype=torch.float32)  # (M,3)
    chi_apo_all  = torch.tensor(chi_apo_all, dtype=torch.float32)  # (M,5)
    chi_holo_all = torch.tensor(chi_holo_all, dtype=torch.float32)  # (M,5)
    chi_mask_all = torch.tensor(chi_mask_all, dtype=torch.int64)  # (M,5)

    return rotations, rot_vecs, translations, chi_apo_all, chi_holo_all, chi_mask_all


chi1_bond_dict = {
    "ALA":None,
    "ARG":("CA", "CB", ["CG", "CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CA", "CB", ["CG", "ND2", "OD1"]),
    "ASP":("CA", "CB", ["CG", "OD1", "OD2"]),
    "CYS":("CA", "CB", ["SG"]),
    "GLN":("CA", "CB", ["CG", "CD", "NE2", "OE1"]),
    "GLU":("CA", "CB", ["CG", "CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CA", "CB", ["CG", "CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CA", "CB", ["CG1", "CG2", "CD1"]),
    "LEU":("CA", "CB", ["CG", "CD1", "CD2"]),
    "LYS":("CA", "CB", ["CG", "CD", "CE", "NZ"]),
    "MET":("CA", "CB", ["CG", "SD", "CE"]),
    "PHE":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CA", "CB", ["CG", "CD"]),
    "SER":("CA", "CB", ["OG"]),
    "THR":("CA", "CB", ["CG2", "OG1"]),
    "TRP":("CA", "CB", ["CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CA", "CB", ["CG", "CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":("CA", "CB", ["CG1", "CG2"])
}

chi2_bond_dict = {
    "ALA":None,
    "ARG":("CB", "CG", ["CD", "NE", "NH1", "NH2", "CZ"]),
    "ASN":("CB", "CG", ["ND2", "OD1"]),
    "ASP":("CB", "CG", ["OD1", "OD2"]),
    "CYS":None,
    "GLN":("CB", "CG", ["CD", "NE2", "OE1"]),
    "GLU":("CB", "CG", ["CD", "OE1", "OE2"]),
    "GLY":None,
    "HIS":("CB", "CG", ["CD2", "ND1", "CE1", "NE2"]),
    "ILE":("CB", "CG1", ["CD1"]),
    "LEU":("CB", "CG", ["CD1", "CD2"]),
    "LYS":("CB", "CG", ["CD", "CE", "NZ"]),
    "MET":("CB", "CG", ["SD", "CE"]),
    "PHE":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "CZ"]),
    "PRO":("CB", "CG", ["CD"]),
    "SER":None,
    "THR":None,
    "TRP":("CB", "CG", ["CD1", "CD2", "CE2", "CE3", "NE1", "CH2", "CZ2", "CZ3"]),
    "TYR":("CB", "CG", ["CD1", "CD2", "CE1", "CE2", "OH", "CZ"]),
    "VAL":None,
}


chi3_bond_dict = {
    "ALA":None,
    "ARG":("CG", "CD", ["NE", "NH1", "NH2", "CZ"]),
    "ASN":None,
    "ASP":None,
    "CYS":None,
    "GLN":("CG", "CD", ["NE2", "OE1"]),
    "GLU":("CG", "CD", ["OE1", "OE2"]),
    "GLY":None,
    "HIS":None,
    "ILE":None,
    "LEU":None,
    "LYS":("CG", "CD", ["CE", "NZ"]),
    "MET":("CG", "SD", ["CE"]),
    "PHE":None,
    "PRO":None,
    "SER":None,
    "THR":None,
    "TRP":None,
    "TYR":None,
    "VAL":None,
}

chi4_bond_dict = {
    "ARG":("CD", "NE", ["NH1", "NH2", "CZ"]),
    "LYS":("CD", "CE", ["NZ"]),
}

chi5_bond_dict = {
    "ARG":("NE", "CZ", ["NH1", "NH2"]),
}

CHI_BOND_DICTS = [
    chi1_bond_dict,   # chi1
    chi2_bond_dict,   # chi2
    chi3_bond_dict,   # chi3
    chi4_bond_dict,   # chi4
    chi5_bond_dict,   # chi5
]

def apply_transforms(
    protein_pos: np.ndarray,                    # (N,3) Apo 坐标
    protein_atom_name: List[str],               # len N
    protein_atom_to_aa_name: List[str],         # len N
    protein_atom_to_aa_group: np.ndarray,       # (N,)
    rotations: np.ndarray,                      # (M,3)  Rodrigues
    translations: np.ndarray,                   # (M,3)
    chi_update: np.ndarray,                     # (M,7) Δχ (rad)
    chi_mask  : np.ndarray                      # (M,7) 0/1
) -> np.ndarray:
    """
    返回 new_pos (N,3) — 先整体刚体变换，再按 Δχ 对侧链原子局部旋转。
    """
    # —— 1. 刚体 —— --------------------------------------------------
    new_pos = protein_pos.copy()
    gid_unique = np.unique(protein_atom_to_aa_group)
    assert len(gid_unique) == rotations.shape[0], "gid 与 R/t 不对应"

    gid2row   = {int(g): i for i, g in enumerate(gid_unique)}
    R_mats    = R.from_rotvec(rotations).as_matrix()          # (M,3,3)

    # for i, gid in enumerate(protein_atom_to_aa_group):
    #     j = gid2row[int(gid)]
    #     new_pos[i] = protein_pos[i] @ R_mats[j].T + translations[j]
    row_idx   = protein_atom_to_aa_group.astype(int)     # (N,)
    R_atom    = R_mats[row_idx]                          # (N,3,3)
    t_atom    = translations[row_idx]                    # (N,3)

    # einsum 可以一次完成 (x @ Rᵀ)
    new_pos = np.einsum('ij,ijk->ik', protein_pos, R_atom) + t_atom

    # —— 2. χ 内旋 —— -------------------------------------------------
    for gid, row in gid2row.items():
        # residue 索引 & 名称
        idxs     = np.where(protein_atom_to_aa_group == gid)[0]
        resname  = protein_atom_to_aa_name[idxs[0]]
        name2idx = {protein_atom_name[k]: k for k in idxs}

        for chi_slot in range(7):
            if not chi_mask[row, chi_slot]:
                continue
            bond_dict = CHI_BOND_DICTS[chi_slot]
            if resname not in bond_dict or bond_dict[resname] is None:
                continue

            atom1, atom2, rot_atoms = bond_dict[resname]
            if atom1 not in name2idx or atom2 not in name2idx:
                continue  # 缺轴原子
            p1 = new_pos[name2idx[atom1]]
            p2 = new_pos[name2idx[atom2]]
            axis = p2 - p1
            norm = np.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis_unit = axis / norm
            theta     = chi_update[row, chi_slot]     # Δχ
            rot_mat   = R.from_rotvec(axis_unit * theta).as_matrix()

            for at in rot_atoms:
                if at not in name2idx:
                    continue
                k  = name2idx[at]
                v  = new_pos[k] - p1
                new_pos[k] = p1 + v @ rot_mat.T

    return new_pos


def quaternion_batch_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of quaternions [N, 4] to rotation matrices [N, 3, 3]
    using numpy-quaternion.
    """
    q_np = quat.detach().cpu().numpy()
    R_list = []
    for q in q_np:
        q_obj = np.quaternion(q[0], q[1], q[2], q[3])
        R = quaternion.as_rotation_matrix(q_obj)
        R_list.append(R)
    return torch.tensor(R_list, dtype=quat.dtype, device=quat.device)


def apply_transforms_tensor(
    protein_pos: torch.Tensor,                # (N,3)  Apo 坐标
    protein_atom_name: List[str],             # len N
    protein_atom_to_aa_name: List[str],       # len N
    protein_atom_to_aa_group: torch.Tensor,   # (N,)  long
    rotations:  torch.Tensor,                 # (M,4) 四元数
    translations: torch.Tensor,               # (M,3)
    chi_update: torch.Tensor,                 # (M,7)
    chi_mask:   torch.Tensor                  # (M,7)
) -> torch.Tensor:
    """
    先整体刚体 (R,t)，再按 Δχ 局部旋转。全部 Torch Tensor，支持 GPU。
    假设 rotations/translations 的行顺序与 gid 一一对应。
    """
    assert protein_pos.ndim == 2 and protein_pos.size(1) == 3
    device = protein_pos.device

    # 1️⃣ 刚体（向量化)
    R_mats = K.geometry.conversions.quaternion_to_rotation_matrix(rotations)  # (M,3,3)
    # 将rotations (M, 3) (rot_vector) 转为旋转矩阵,这里的输入不再是四元数，而是旋转向量
    # R_mats = K.geometry.conversions.axis_angle_to_rotation_matrix(rotations)      # (M,3,3)
    row_idx = protein_atom_to_aa_group                              # (N,)
    new_pos = (
        torch.einsum('ni,nij->nj', protein_pos, R_mats[row_idx].transpose(-1, -2))
        + translations[row_idx]
    )

    # 2️⃣ χ 内旋
    gid_unique = torch.unique(row_idx).tolist()
    for row, gid in enumerate(gid_unique):
        idxs = (row_idx == gid).nonzero(as_tuple=True)[0]
        resname = protein_atom_to_aa_name[idxs[0].item()]
        name2idx = {protein_atom_name[i.item()]: i.item() for i in idxs}

        for chi_slot in range(5):
            if chi_mask[row, chi_slot] == 0:
                continue
            bond_dict = CHI_BOND_DICTS[chi_slot]
            if resname not in bond_dict or bond_dict[resname] is None:
                continue
            atom1, atom2, rot_atoms = bond_dict[resname]
            if atom1 not in name2idx or atom2 not in name2idx:
                continue

            p1 = new_pos[name2idx[atom1]]
            p2 = new_pos[name2idx[atom2]]
            axis = p2 - p1
            norm = torch.linalg.norm(axis)
            if norm < 1e-6:
                continue
            axis_unit = axis / norm
            theta     = chi_update[row, chi_slot]
            rot_mat   = K.geometry.conversions.axis_angle_to_rotation_matrix((axis_unit * theta).unsqueeze(0))[0]

            for at in rot_atoms:
                if at not in name2idx: continue
                k = name2idx[at]
                v = new_pos[k] - p1
                new_pos[k] = p1 + v @ rot_mat.T

    return new_pos


def apply_transforms_tensor_batch(
    protein_pos: torch.Tensor,                # (N,3)  所有蛋白的原子坐标拼接
    protein_atom_name: List[List[str]],       # len = #proteins, 每条子表是该蛋白所有原子名
    protein_atom_to_aa_name: List[List[str]], # len = #proteins, 每条子表是该蛋白残基名
    protein_atom_to_aa_group: torch.Tensor,   # (N,)   原子→局部残基 id，按蛋白各自从 0 计数
    protein_element_batch: torch.Tensor,      # (N,)   每个原子属于哪条蛋白
    rotations: torch.Tensor,                  # (M,4)  所有残基旋转四元数拼接
    translations: torch.Tensor,               # (M,3)  所有残基平移向量拼接
    chi_update: torch.Tensor,                 # (M,5)  χ 角增量
    chi_mask: torch.Tensor,                   # (M,5)  χ 角是否更新
    protein_translations_batch: torch.Tensor, # (M,)   每个残基属于哪条蛋白
) -> torch.Tensor:
    """
    Batched version of `apply_transforms_tensor`.

    * `protein_element_batch[i]` 给出第 i 个原子对应的蛋白 id。
    * `protein_translations_batch[j]` 给出第 j 个残基 (即 rotations[j]) 对应的蛋白 id。
    * `protein_atom_to_aa_group` 在 **每条蛋白内部** 都是 0,1,2,... 重新编号。
    返回值顺序与输入 atoms 顺序一致。
    """
    device = protein_pos.device
    new_pos = protein_pos.clone()

    num_proteins = len(protein_atom_name)
    assert num_proteins == len(protein_atom_to_aa_name), "两份列表长度应一致"

    # 逐条蛋白处理：掩码切片 → 调用单蛋白版本 → 回填
    for p_idx in range(num_proteins):
        # atoms 属于这条蛋白的布尔掩码
        atom_mask = (protein_element_batch.squeeze(-1) == p_idx)
        if atom_mask.sum() == 0:
            continue

        # residues 属于这条蛋白的布尔掩码
        resid_mask = (protein_translations_batch.squeeze(-1) == p_idx)

        # --- 切出当前蛋白的数据 ---
        pos_p            = protein_pos[atom_mask]                         # (N_p,3)
        atom_name_p      = protein_atom_name[p_idx]                       # List[str]
        aa_name_p        = protein_atom_to_aa_name[p_idx]                 # List[str]
        aa_group_p       = protein_atom_to_aa_group[atom_mask]            # (N_p,)
        rotations_p      = rotations[resid_mask]                          # (M_p,4)
        translations_p   = translations[resid_mask]                       # (M_p,3)
        chi_update_p     = chi_update[resid_mask]                         # (M_p,5)
        chi_mask_p       = chi_mask[resid_mask]                           # (M_p,5)

        # --- 调用单蛋白函数 ---
        new_pos_p = apply_transforms_tensor(
            pos_p,
            atom_name_p,
            aa_name_p,
            aa_group_p,
            rotations_p,
            translations_p,
            chi_update_p,
            chi_mask_p,
        )

        # --- 把结果写回到总张量 ---
        new_pos[atom_mask] = new_pos_p

    return new_pos


class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        self.atom_to_aa_name = []
        self.atom_to_aa_group = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])
            self.atom_to_aa_name.append(atom['res_name'])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)
        # group_number = 0
        # start_aa_type = self.atom_to_aa_type[0]
        # for i, aa_type in enumerate(self.atom_to_aa_type):
        #     if aa_type != start_aa_type:
        #         group_number += 1
        #         start_aa_type = aa_type
        #     self.atom_to_aa_group.append(group_number)
        atom_to_residue_group = {}
        for group_idx, residue in enumerate(residues_tmp.values()):
            for atom_idx in residue['atoms']:
                atom_to_residue_group[atom_idx] = group_idx

        # Build the atom_to_aa_group list in the correct order
        for i in range(len(self.atom_to_aa_type)):
            self.atom_to_aa_group.append(atom_to_residue_group[i])

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.int64),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool_),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.int64),
            'atom_to_aa_name': self.atom_to_aa_name,
            'atom_to_aa_group': np.array(self.atom_to_aa_group, dtype=np.int64),
            # 'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            # 'pos_C': np.array(self.pos_C, dtype=np.float32),
            # 'pos_N': np.array(self.pos_N, dtype=np.float32),
            # 'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.int64),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int64)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.int64)
    edge_type = np.array(edge_type, dtype=np.int64)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data
