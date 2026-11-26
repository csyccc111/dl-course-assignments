# dl-course-assignments

深圳技术大学《深度学习方法与应用》（IB00267）课程实验代码归档。

## 实验

- [exp7](exp7/README.md)：人脸识别系统设计。YuNet / CenterFace 检测、evoLVe 5 点对齐、SFace 特征、1:1 / 1:N 评测、视频演示。
- [exp8](exp8/README.md)：ChnSentiCorp 评论情感分类。RNN / LSTM / Transformer 从零训练 + 预训练 BERT 微调；云端运行见 [exp8/README-cloud.md](exp8/README-cloud.md)。

## 说明

- 数据集、ONNX 模型、运行产物和课程报告未入库，避免仓库过大和公开个人信息；各实验 README 说明数据的获取或生成方式。
- exp8 的 ChnSentiCorp 三个 tsv 已入库（约 3.7 MB），云端可直接 clone 运行。
- 实验 7 的 LFW / CASIA-WebFace、ONNX 模型和 `outputs/` 按 `exp7/.gitignore` 排除，需要时按 `exp7/README.md` / `exp7/frontend/README.md` 重新下载或生成。