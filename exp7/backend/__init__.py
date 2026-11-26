"""实验七 B 角色：人脸识别与人脸库。

模块：
- embedder: YuNet 检测 + SFace 对齐/特征提取
- build_embeddings: 批量提取 embedding
- eval_lfw: LFW 1:1 验证评测
- gallery: 人脸库注册/匹配
- train_head: 在冻结 embedding 上训练分类头
"""
