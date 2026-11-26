# -*- coding: utf-8 -*-
"""
align_trans.py —— 人脸 5 点相似变换对齐（改编自 face.evoLVe）
================================================================

来源
----
https://github.com/ZhaoJ9014/face.evoLVe/tree/master/applications/align
    ├── align_trans.py      -> 本文件主体
    ├── matlab_cp2tform.py  -> 已合并进本文件（Umeyama 相似变换 / MATLAB cp2tform 移植）
    └── face_align.py       -> 仅参考其调用方式（get_reference_facial_points * scale + warp_and_crop_face）

原仓库代码是 Python2 时代遗留，直接拿来跑有若干问题，本文件的改动如下
（对应报告「踩坑记录」章节，可直接引用）：

1. `if align_type is 'cv2_affine'` —— 用 `is` 比较字符串。
   Python3.8+ 抛 SyntaxWarning，3.12 起语义不再保证。改为 `==`。

2. 异常类 `__str__` 里写 `super.__str__(self)` —— `super` 是内建类型而非实例方法，
   一旦真的触发异常会再抛 AttributeError，把原始错误信息吞掉。重写为正常实现。

3. `findSimilarity()` 中 `xyR = xy` 是**引用**而不是拷贝：
   紧接着 `xyR[:, 0] = -1 * xyR[:, 0]` 会就地翻转调用方传入的目标点，
   导致后面 `norm1 = norm(xy1 - xy)` / `norm2 = norm(xy2 - xy)` 全都在拿
   被污染过的 xy 做比较，反射/非反射两路的选择逻辑实际是错的。改为 `xy.copy()`。

4. `np.linalg.lstsq(A, b)` 未传 rcond，numpy>=1.14 会告警（FutureWarning），
   且新版本默认行为会变。显式传 `rcond=None`。

5. `warp_and_crop_face()` 在 `crop_size != (96, 112)` 且 `reference_pts=None` 时
   必定抛 FaceWarpException（因为 get_reference_facial_points 在
   inner_padding_factor==0 且 outer_padding==(0,0) 且 output_size 与默认裁剪尺寸不等时
   直接 raise）。改为统一走 `build_reference()`，128/160 等尺寸也能用。

6. 新增 `reflective` 开关，**默认 False**。
   原实现 `reflective=True` 会允许「镜像相似变换」——对镜像解来说残差同样为 0，
   于是「左右眼顺序搞反」和「正常顺序」在数值上无法区分，对齐出来的人脸是左右翻的，
   送进识别模型后特征会明显偏移。人脸对齐不应该存在镜像解，故默认关闭。

7. 新增 `similarity_residual()`：返回最小二乘相似变换的拟合残差（RMS，单位：像素）。
   它同时是「关键点顺序自检」的判据和「对齐质量」的量化指标。

坐标约定
--------
REFERENCE_FACIAL_POINTS 的顺序为：
    0 = 图像左侧眼睛（被摄者的右眼）
    1 = 图像右侧眼睛（被摄者的左眼）
    2 = 鼻尖
    3 = 图像左侧嘴角
    4 = 图像右侧嘴角
即 ArcFace 112x112 标准 5 点模板。
"""

from __future__ import annotations

import numpy as np
import cv2
from numpy.linalg import inv, lstsq
from numpy.linalg import matrix_rank as _matrix_rank


# 默认参考关键点：crop_size = (96, 112) 时的标准 5 点
REFERENCE_FACIAL_POINTS = np.array(
    [
        [30.29459953, 51.69630051],
        [65.53179932, 51.50139999],
        [48.02519989, 71.73660278],
        [33.54930115, 92.36550140],
        [62.72990036, 92.20410156],
    ],
    dtype=np.float32,
)

DEFAULT_CROP_SIZE = (96, 112)

# 112x112 正方形模板的基准边长（build_reference 里按比例缩放用）
_BASE_SQUARE = 112.0


class FaceWarpException(Exception):
    """对齐过程中的参数或数据异常。"""


# ---------------------------------------------------------------------------
# 基础变换工具（源自 matlab_cp2tform.py）
# ---------------------------------------------------------------------------
def tformfwd(trans, uv):
    """把 3x3 仿射矩阵 trans 作用到 Kx2 坐标 uv 上，返回 Kx2。"""
    uv = np.asarray(uv, dtype=np.float64)
    uv_h = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv_h, trans)
    return xy[:, 0:-1]


def tforminv(trans, uv):
    """应用 trans 的逆变换。"""
    return tformfwd(inv(trans), uv)


def find_nonreflective_similarity(uv, xy, options=None):
    """求解非反射相似变换（MATLAB cp2tform 'similarity'）。

    求解 `[u, v, 1] @ trans = [x, y, 1]`，其中 (u,v) 来自 uv，(x,y) 来自 xy。
    返回 (trans, trans_inv)，均为 3x3。
    """
    options = {"K": 2}
    K = options["K"]

    xy = np.asarray(xy, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if _matrix_rank(X) >= 2 * K:
        # rcond=None：numpy>=1.14 的推荐写法，避免 FutureWarning
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise FaceWarpException("cp2tform 需要至少 2 组不重合的点")

    sc, ss, tx, ty = r[0], r[1], r[2], r[3]

    Tinv = np.array(
        [
            [sc, -ss, 0.0],
            [ss, sc, 0.0],
            [tx, ty, 1.0],
        ],
        dtype=np.float64,
    )

    T = inv(Tinv)
    T[:, 2] = np.array([0.0, 0.0, 1.0])
    return T, Tinv


def find_similarity(uv, xy, options=None):
    """在反射 / 非反射两个解里挑残差更小的那个。"""
    trans1, trans1_inv = find_nonreflective_similarity(uv, xy, options)

    # 原版此处 `xyR = xy` 会就地污染 xy，这里必须拷贝
    xyR = np.array(xy, dtype=np.float64, copy=True)
    xyR[:, 0] = -1.0 * xyR[:, 0]
    trans2r, _ = find_nonreflective_similarity(uv, xyR, options)

    TreflectY = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    trans2 = np.dot(trans2r, TreflectY)

    xy1 = tformfwd(trans1, uv)
    norm1 = np.linalg.norm(xy1 - xy)

    xy2 = tformfwd(trans2, uv)
    norm2 = np.linalg.norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    return trans2, inv(trans2)


# 兼容原仓库命名
findNonreflectiveSimilarity = find_nonreflective_similarity
findSimilarity = find_similarity


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """求解把 src_pts 映射到 dst_pts 的相似变换，返回 (trans, trans_inv) 两个 3x3。

    注意：人脸对齐建议传 reflective=False，避免出现镜像解（见文件头说明 6）。
    """
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    if src_pts.shape != dst_pts.shape:
        raise FaceWarpException(
            "src_pts 与 dst_pts 形状必须一致，当前为 {} vs {}".format(
                src_pts.shape, dst_pts.shape
            )
        )
    if reflective:
        return find_similarity(src_pts, dst_pts)
    return find_nonreflective_similarity(src_pts, dst_pts)


def cvt_tform_mat_for_cv2(trans):
    """3x3（行向量约定）-> cv2.warpAffine 需要的 2x3。"""
    return np.asarray(trans, dtype=np.float32)[:, 0:2].T


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """直接返回可给 cv2.warpAffine 使用的 2x3 矩阵。"""
    trans, _ = get_similarity_transform(src_pts, dst_pts, reflective)
    return cvt_tform_mat_for_cv2(trans)


def similarity_residual(src_pts, dst_pts, trans):
    """相似变换的最小二乘拟合残差（RMS，像素）。

    参数
    ----
    src_pts, dst_pts : Kx2
    trans            : 3x3，`[u, v, 1] @ trans = [x, y, 1]`

    返回
    ----
    float，单位为「目标坐标系下的像素」。对齐时目标坐标系就是 112x112 裁剪图，
    因此数值可直接读作「关键点平均偏差多少个像素」。
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    proj = tformfwd(trans, src)
    return float(np.sqrt(np.mean(np.sum((proj - dst) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# 参考点构造
# ---------------------------------------------------------------------------
def get_reference_facial_points(
    output_size=None,
    inner_padding_factor=0.0,
    outer_padding=(0, 0),
    default_square=False,
):
    """按裁剪设置生成参考 5 点（保持与原仓库一致的语义）。

    default_square=True 时以 (112,112) 为基准，此时返回的就是 ArcFace 标准模板：
        [[38.29, 51.70], [73.53, 51.50], [56.03, 71.74], [41.55, 92.37], [70.73, 92.20]]
    """
    tmp_5pts = REFERENCE_FACIAL_POINTS.astype(np.float64).copy()
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE, dtype=np.float64)

    # 0) 把内部区域补成正方形
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2.0
        tmp_crop_size += size_diff

    if output_size is not None:
        output_size = np.array(output_size, dtype=np.float64)
        if output_size[0] == tmp_crop_size[0] and output_size[1] == tmp_crop_size[1]:
            return tmp_5pts.astype(np.float32)

    if inner_padding_factor == 0 and tuple(outer_padding) == (0, 0):
        if output_size is None:
            return tmp_5pts.astype(np.float32)
        raise FaceWarpException(
            "没有 padding 需要处理，output_size 必须是 None 或 {}".format(tmp_crop_size)
        )

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException("要求 0 <= inner_padding_factor <= 1.0")

    if (
        inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0
    ) and output_size is None:
        output_size = (tmp_crop_size * (1 + inner_padding_factor * 2)).astype(np.int32)
        output_size = output_size + np.array(outer_padding, dtype=np.int32)
        output_size = output_size.astype(np.float64)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException("outer_padding 必须小于 output_size")

    # 1) 按 inner_padding_factor 内缩
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2.0
        tmp_crop_size += np.round(size_diff)

    # 2) 缩放到 (output_size - 2*outer_padding)
    size_bf_outer_pad = output_size - np.array(outer_padding, dtype=np.float64) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException(
            "必须满足 (output_size - outer_padding) = scale * (crop_size * (1 + inner_padding_factor))"
        )

    scale_factor = float(size_bf_outer_pad[0] / tmp_crop_size[0])
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad

    # 3) 加上外框 padding
    reference_5point = tmp_5pts + np.array(outer_padding, dtype=np.float64)
    return reference_5point.astype(np.float32)


def build_reference(crop_size=(112, 112)):
    """根据裁剪尺寸返回参考 5 点。

    只支持正方形输出（112x112 / 128x128 / 160x160 ...），这是人脸识别的通行做法；
    等价于原仓库 face_align.py 里的
        scale = crop_size / 112; reference = get_reference_facial_points(default_square=True) * scale
    但把「crop_size != 96x112 且 reference_pts=None 就抛异常」的老问题一并修掉了。
    """
    w, h = int(crop_size[0]), int(crop_size[1])
    if (w, h) == (96, 112):
        return REFERENCE_FACIAL_POINTS.copy()
    if w != h:
        raise ValueError("build_reference 目前只支持正方形裁剪尺寸，收到 {}".format(crop_size))

    base = get_reference_facial_points(default_square=True)  # 112x112
    return (base * (w / _BASE_SQUARE)).astype(np.float32)


# ---------------------------------------------------------------------------
# 裁剪
# ---------------------------------------------------------------------------
def warp_and_crop_face(
    src_img,
    facial_pts,
    reference_pts=None,
    crop_size=(96, 112),
    align_type="similarity",
    reflective=False,
):
    """按参考点把原图 warp 成对齐人脸。

    参数
    ----
    src_img       : HxWx3 BGR
    facial_pts    : Kx2 或 2xK（K>=3），原始图像坐标系
    reference_pts : Kx2 或 2xK，None 时用 build_reference(crop_size)
    crop_size     : (w, h)
    align_type    : 'similarity' | 'affine' | 'cv2_affine'
    reflective    : 是否允许镜像相似变换（默认 False）

    返回
    ----
    face_img : crop_size[1] x crop_size[0] x 3
    """
    if reference_pts is None:
        reference_pts = build_reference(crop_size)

    ref_pts = np.float32(reference_pts)
    if max(ref_pts.shape) < 3 or min(ref_pts.shape) != 2:
        raise FaceWarpException("reference_pts 形状必须是 (K,2) 或 (2,K) 且 K>=3")
    if ref_pts.shape[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    if max(src_pts.shape) < 3 or min(src_pts.shape) != 2:
        raise FaceWarpException("facial_pts 形状必须是 (K,2) 或 (2,K) 且 K>=3")
    if src_pts.shape[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException("facial_pts 与 reference_pts 形状必须一致")

    if align_type == "cv2_affine":          # 原版此处用 `is`，已修正
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == "affine":
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    else:
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts, reflective)

    face_img = cv2.warpAffine(
        src_img,
        tfm,
        (int(crop_size[0]), int(crop_size[1])),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return face_img


def get_affine_transform_matrix(src_pts, dst_pts):
    """全点最小二乘仿射（对应 align_type='affine'）。"""
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    n_pts = src_pts.shape[0]

    src_pts_ = np.hstack([src_pts, np.ones((n_pts, 1))])
    dst_pts_ = np.hstack([dst_pts, np.ones((n_pts, 1))])

    A, _, rank, _ = lstsq(src_pts_, dst_pts_, rcond=None)

    if rank == 3:
        tfm = np.float32(
            [
                [A[0, 0], A[1, 0], A[2, 0]],
                [A[0, 1], A[1, 1], A[2, 1]],
            ]
        )
    elif rank == 2:
        tfm = np.float32(
            [
                [A[0, 0], A[1, 0], 0.0],
                [A[0, 1], A[1, 1], 0.0],
            ]
        )
    else:
        raise FaceWarpException("点集退化为 rank={}，无法求解仿射矩阵".format(rank))
    return tfm
