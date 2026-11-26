"""在冻结的 SFace embedding 上训练分类头。

这是 B 的第一版训练脚本：不训练 CNN，只在已缓存的 embedding 上训练
LogisticRegression / MLP，快速验证数据、标签和评估流程。
端到端微调等 baseline 稳定后再做。

用法：
  python -m backend.train_head \
      --embeddings outputs/self.npz \
      --output outputs/head_logreg.pkl
"""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from .paths import resolve
except ImportError:  # 允许 python backend/train_head.py 直接运行
    from paths import resolve


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 embedding 分类头")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default="outputs/head.pkl")
    parser.add_argument("--metrics", default="outputs/head_metrics.json")
    parser.add_argument("--model", choices=["logreg", "mlp"], default="logreg")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--C", type=float, default=1.0, help="LogisticRegression 正则强度")
    args = parser.parse_args()

    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import train_test_split
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import LabelEncoder
    except ImportError as exc:
        raise SystemExit(
            "需要 scikit-learn/joblib：pip install scikit-learn joblib"
        ) from exc

    data = np.load(resolve(args.embeddings), allow_pickle=False)
    features = data["embeddings"].astype(np.float32)
    raw_labels = data["labels"].astype(str)
    splits = data["splits"].astype(str) if "splits" in data else None

    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    if len(encoder.classes_) < 2:
        raise SystemExit("至少需要 2 个身份才能训练分类头")

    split_mode = "random"
    if splits is not None:
        available = set(splits.tolist())
        train_mask = None
        for train_name in ("train", "enroll"):
            if train_name in available:
                train_mask = splits == train_name
                break
        test_mask = splits == "test" if "test" in available else None

        if (
            train_mask is not None
            and test_mask is not None
            and train_mask.any()
            and test_mask.any()
        ):
            split_mode = "preset"
            x_train, y_train = features[train_mask], labels[train_mask]
            x_test, y_test = features[test_mask], labels[test_mask]
            print("使用数据自带的 train/enroll + test 划分")
        else:
            print("未找到 train/enroll + test 划分，改用随机划分")

    if split_mode == "random":
        counts = np.bincount(labels)
        stratify = labels if counts.min() >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify,
        )

    if len(np.unique(y_train)) < 2:
        raise SystemExit("训练集只有一个身份，无法训练")
    if len(np.unique(y_test)) < 2:
        print("警告：测试集只有一个身份，准确率参考意义有限")

    print(
        f"训练样本: {len(x_train)}  测试样本: {len(x_test)}  "
        f"身份数: {len(encoder.classes_)}"
    )
    print("注意：这是 closed-set 分类头，测试身份与训练身份相同；"
          "开放集识别请用人脸库 + 阈值。")

    if args.model == "logreg":
        classifier = LogisticRegression(max_iter=2000, C=args.C)
    else:
        classifier = MLPClassifier(
            hidden_layer_sizes=(256,),
            max_iter=500,
            random_state=args.seed,
            early_stopping=True,
            n_iter_no_change=20,
        )

    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    report = classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_test, predictions, labels=np.arange(len(encoder.classes_))
    )

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": classifier,
            "label_encoder": encoder,
            "model_type": args.model,
            "input_dim": int(features.shape[1]),
            "embedding_model": "SFace face_recognition_sface_2021dec.onnx",
        },
        output,
    )

    metrics = {
        "accuracy": accuracy,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "n_identities": int(len(encoder.classes_)),
        "model": args.model,
        "split_mode": split_mode,
        "classification_report": report,
        "confusion_matrix": matrix,
        "classes": encoder.classes_.tolist(),
    }
    metrics_path = resolve(args.metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"测试准确率: {accuracy:.4f}")
    print(f"模型: {output}")
    print(f"指标: {metrics_path}")


if __name__ == "__main__":
    main()
