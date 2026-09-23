# Blind PRNU-based Image Clustering Pipeline

カメラの固有ノイズ（PRNU: Photo-Response Non-Uniformity）を利用して、撮影元カメラごとに画像を自動分類（クラスタリング）する Python パイプラインです。  
指定した複数のクラスタ数（$k = 5, 10, 15, 20$）で自動的に処理を実行し、結果を CSV および TXT ファイル形式で保存します。

---

## 概要

本プロジェクトは、カメラの個別識別技術である PRNU ノイズ残差の類似度計算とアンサンブルクラスタリングを組み合わせた画像分類ツールです。以下の3ステップで処理を行います。

1. **特徴抽出と内積マトリクスの計算 (Step 1)**  
   BM3D などのノイズ除去フィルタで画像から PRNU ノイズ残差を抽出し、一定サイズ（例: 500x500）にクロップした領域間の類似度行列 $C$ を計算します。
2. **初期クラスタリング (Step 2)**  
   AL-ICM（エネルギー最適化）を用いて複数の初期ベースクラスタを生成し、WEAC（アンサンブル統合）で1つのクラスタ結果に統合します。
3. **洗練ステップ (Step 3)**  
   生の内積類似度に基づき、Iterative Refinement で境界部分のラベルを微調整して最終結果を決定します。

---

## 特徴

- **複数のクラスタ数（$k$）の一括評価**: 異なる想定カメラ数（$k = 5, 10, 15, 20$ 等）でのクラスタリングを連続実行。
- **処理の最適化**: 最も時間のかかる PRNU 抽出・類似度行列計算（Step 1）を1度だけ実行し、以降のステップで再利用。
- **結果の自動保存**: 実行結果は `clustering_output_results/` ディレクトリ内に CSV（データ解析用）および TXT（閲覧用）として自動出力。

---

## ディレクトリ構成

```text
.
├── main.py                     # メイン実行スクリプト
├── clustering_utils.py         # コアアルゴリズム関数群
└── clustering_output_results/  # 実行結果保存先（自動生成）
    ├── clustering_k5.csv
    ├── clustering_k5.txt
    ├── clustering_k10.csv
    └── ...
```

---

## 動作要件・依存ライブラリ

- **Python**: 3.8 以上
- **主なパッケージ**:
  - `numpy`
  - `pillow`
  - `tqdm`

### インストールコマンド

```bash
pip install numpy pillow tqdm
```

> **注意**: `clustering_utils.py` の内部実装（BM3D等のノイズ除去アルゴリズム）に応じて、追加のC++ライブラリやライブラリのセットアップが必要になる場合があります。

---

## 使い方

### 1. 設定の変更
`main.py` 内の以下の部分を環境に合わせて書き換えます。

```python
# ★分類対象の画像が入っているディレクトリを指定
dataset_dir = "/Users/username/Desktop/fashion-pic"

# ★比較評価したいクラスタ数のリスト
target_k_list = [5, 10, 15, 20]
```

### 2. パイプラインの実行
ターミナルからスクリプトを実行します。

```bash
python main.py
```

---

## 出力フォーマット

処理完了後、`clustering_output_results/` フォルダ内に設定した $k$ ごとの結果が出力されます。

### 1. CSV ファイル (`clustering_k{k}.csv`)
精度評価やデータ解析用のフォーマットです。

| image_path | filename | cluster_id |
| :--- | :--- | :--- |
| `/path/to/img01.jpg` | `img01.jpg` | `0` |
| `/path/to/img02.jpg` | `img02.jpg` | `1` |

### 2. テキストファイル (`clustering_k{k}.txt`)
閲覧・グループ確認用のテキストフォーマットです。

```text
=== PRNU Clustering Results (Target k = 5) ===

[カメラグループ 0]
  - img01.jpg
  - img05.jpg

[カメラグループ 1]
  - img02.jpg
  - img03.jpg
```

---

## スクリプト一覧

<details>
<summary><b>メインスクリプト (main.py) のコードを表示</b></summary>

```python
import os
import glob
import csv
import numpy as np

from clustering_utils import (
    ResidualInnerProductMatrix,  # Step 1用
    al_icm,                      # Step 2用 (エネルギー最適化)
    weac_python,                 # Step 2用 (アンサンブル統合)
    refine_clustering_labels     # Step 3用 (洗練ステップ)
)

def main():
    print("=== Blind PRNU-based Image Clustering Pipeline ===")

    # ---------------------------------------------------------
    # [準備] 画像データの読み込み
    # ---------------------------------------------------------
    dataset_dir = "/Users/qingfengliu/Desktop/fashion-pic"
    images_name = sorted(glob.glob(os.path.join(dataset_dir, "*.jpg")) + glob.glob(os.path.join(dataset_dir, "*.png")))

    n_samples = len(images_name)
    if n_samples == 0:
        print("エラー: 指定されたディレクトリに画像が見つかりません。")
        return

    print(f"対象画像数: {n_samples}枚")

    output_dir = "./clustering_output_results"
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================
    # Step 1: 特徴抽出と内積マトリクスの計算 (1度のみ実行)
    # =========================================================
    print("\n--- Step 1: ノイズ残差の抽出と内積マトリクス(C)の計算 ---")
    C = ResidualInnerProductMatrix(
        images_name, 
        denoising_function='bm3d',
        crop_size=(500, 500)
    )

    threshold = np.mean(C)
    W = C - threshold

    target_k_list = [5, 10, 15, 20]

    # =========================================================
    # Step 2 & 3: 各クラスタ数ごとの処理と保存
    # =========================================================
    for k in target_k_list:
        print(f"\n==========================================")
        print(f"  クラスタ数 k = {k} の処理を開始")
        print(f"==========================================")

        # Step 2: 初期クラスタリング
        base_cls_list = []
        for seed in range(5):
            np.random.seed(seed)
            random_init = np.random.randint(0, k, n_samples)
            opt_labels = al_icm(W, initial_labels=random_init)
            base_cls_list.append(opt_labels)

        base_cls = np.column_stack(base_cls_list)
        dummy_ncai = np.ones(base_cls.shape[1])

        weac_results = weac_python(base_cls, dummy_ncai, cls_nums=[k])
        
        if isinstance(weac_results, dict) and k in weac_results:
            initial_labels = weac_results[k]
        elif isinstance(weac_results, dict) and len(weac_results) > 0:
            initial_labels = list(weac_results.values())[0]
        else:
            initial_labels = weac_results

        # Step 3: 洗練ステップ
        final_labels = refine_clustering_labels(C, initial_labels, max_iter=20)

        # 結果の出力 & ファイル保存
        unique_cameras = np.unique(final_labels)
        
        csv_filename = os.path.join(output_dir, f"clustering_k{k}.csv")
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "filename", "cluster_id"])
            for path, label in zip(images_name, final_labels):
                writer.writerow([path, os.path.basename(path), label])

        txt_filename = os.path.join(output_dir, f"clustering_k{k}.txt")
        with open(txt_filename, mode="w", encoding="utf-8") as f:
            f.write(f"=== PRNU Clustering Results (Target k = {k}) ===\n")
            for cam_id in unique_cameras:
                f.write(f"\n[カメラグループ {cam_id}]\n")
                for i, path in enumerate(images_name):
                    if final_labels[i] == cam_id:
                        f.write(f"  - {os.path.basename(path)}\n")

    print("\n=== すべてのクラスタ数での処理が完了しました ===")

if __name__ == "__main__":
    main()
```

</details>
