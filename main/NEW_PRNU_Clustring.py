import csv
from collections import defaultdict
import io
import os
import random
import zipfile
import numpy as np
from PIL import Image
from sklearn.metrics import adjusted_rand_score, rand_score, normalized_mutual_info_score

# リファクタリングしたモジュールから関数をインポート
from clustering_utils_new_2 import (
    ResidualInnerProductMatrix,   # Step 1: ノイズ残差抽出 & 内積マトリクス計算
    generate_base_clusterings,    # Step 2-1: alpha スウィープによるベースクラスタリング生成
    weac_consensus_clustering,   # Step 2-2: WEAC 統合
    refine_clustering_paper,     # Step 3: 対数尤度比 Lambda_pq による精錬ステップ
)


def is_image_corrupted(zip_ref, filename):
    """
    ZIP内の画像バイナリを読み込み、PILで正常にデコードできるか検証する
    """
    try:
        file_bytes = zip_ref.read(filename)
        if not file_bytes or len(file_bytes) < 100:
            return True

        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()

        return False  # 正常
    except Exception:
        return True  # 破損またはデコード不可


def main():
    print("=== PRNU-Based Image Clustering Pipeline (Marra et al., 2017) ===")

    # =========================================================
    # 設定パラメータ
    # =========================================================
    zip_file_path = "/Users/qingfengliu/Desktop/archive.zip"  # ZIPファイルのパス
    NUM_FOLDERS = 5            # 使用するフォルダ（カメラ）数
    RANDOM_SEED = 42           # ランダム抽出の再現用シード（毎回変更したい場合は None）
    DENOISING_FUNC = "wavelet" # ノイズ除去フィルタ ('wavelet' または 'bm3d')
    CROP_SIZE = (500, 500)     # 画像切り出しサイズ (縦, 横)

    # ★ クラスタ数 k の指定設定
    # 任意の値（例: 5）を指定するとそのクラスタ数に分割します。
    # None に設定すると論文通りの自動判定モード（l* 自動決定）になります。
    TARGET_K = 5

    if not os.path.exists(zip_file_path):
        raise FileNotFoundError(f"ZIPファイルが見つかりません: {zip_file_path}")

    # ---------------------------------------------------------
    # 1. ZIPファイルをスキャンし、フォルダごとに画像を分類
    # ---------------------------------------------------------
    print(f"\nZIPファイルをスキャン＆フォルダ構造解析中: {zip_file_path}")
    valid_extensions = (".jpg", ".jpeg", ".png")
    folder_to_images = defaultdict(list)
    corrupted_count = 0

    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        all_files = zip_ref.infolist()
        print(f"総ファイル数: {len(all_files)} 件")

        for file_info in all_files:
            filename = file_info.filename

            if (
                file_info.is_dir()
                or "__MACOSX" in filename
                or os.path.basename(filename).startswith(".")
            ):
                continue

            if filename.lower().endswith(valid_extensions):
                if is_image_corrupted(zip_ref, filename):
                    corrupted_count += 1
                    continue

                folder_path = os.path.dirname(filename)
                folder_to_images[folder_path].append(filename)

    available_folders = sorted(list(folder_to_images.keys()))
    total_folders = len(available_folders)

    print("\n" + "=" * 50)
    print("【事前スキャン結果】")
    print(f"  ・検出されたフォルダ（クラス）数: {total_folders} 個")
    print(f"  ・事前に除外された破損画像数    : {corrupted_count} 枚")
    print("=" * 50)

    if total_folders == 0:
        raise ValueError("処理可能な正常画像を含むフォルダが見つかりませんでした。")

    # ---------------------------------------------------------
    # 2. フォルダ単位でのサンプリング
    # ---------------------------------------------------------
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    select_folder_count = min(NUM_FOLDERS, total_folders)
    selected_folders = random.sample(available_folders, select_folder_count)

    images_name = []
    print(f"\n【実行対象フォルダの選択（全{total_folders}個中 {select_folder_count}個を抽出）】")
    for f in selected_folders:
        imgs = folder_to_images[f]
        images_name.extend(imgs)
        print(f"  ・フォルダ: {f} （{len(imgs)}枚）")

    images_name = sorted(images_name)
    print("-" * 50)
    print(f"  抽出された初期画像数: {len(images_name)} 枚")
    print("=" * 50 + "\n")

    # =========================================================
    # Step 1: ノイズ残差の抽出と内積マトリクス (C) の計算
    # =========================================================
    print("--- Step 1: ノイズ残差抽出と内積マトリクス(C)の計算 ---")
    
    C = ResidualInnerProductMatrix(
        images_name,
        zip_file_path=zip_file_path,
        denoising_function=DENOISING_FUNC,
        crop_size=CROP_SIZE,
        crop_location="center",
    )

    n_samples = len(images_name)

    ground_truth_folders = [
        os.path.basename(os.path.dirname(p)) for p in images_name
    ]
    unique_gt_names, ground_truth_labels = np.unique(
        ground_truth_folders, return_inverse=True
    )

    print(f"\n[前処理完了後の有効データ]")
    print(f"  有効画像数           : {n_samples} 枚")
    print(f"  Ground Truth クラス数: {len(unique_gt_names)} 個 ({list(unique_gt_names)})")

    # 指定されたTARGET_Kの確認
    if TARGET_K is not None:
        print(f"  設定された指定クラスタ数 k : {TARGET_K} 個")
    else:
        print(f"  設定された指定クラスタ数 k : 自動判定（Auto）")

    # =========================================================
    # Step 2 & 3: クラスタリング実行
    # =========================================================
    print("\n" + "=" * 50)
    print("  クラスタリング処理を開始")
    print("=" * 50)

    # --- Step 2-1: alpha スウィープによる相関クラスタリング ---
    print("\n--- Step 2-1: Correlation Clustering (alpha スウィープ) ---")
    base_partitions = generate_base_clusterings(C, num_alphas=41)

    # --- Step 2-2: WEAC アンサンブル統合 ---
    print("\n--- Step 2-2: WEAC 合意クラスタリング ---")
    initial_labels = weac_consensus_clustering(
        base_partitions, target_k=TARGET_K, epsilon=0.01
    )

    # --- Step 3: 対数尤度比 (Lambda_pq) による精錬 (Refinement) ---
    print("\n--- Step 3: 対数尤度比 (Lambda_pq) による精錬ステップ ---")
    final_labels = refine_clustering_paper(C, initial_labels)

    # =========================================================
    # 評価指標の計算 (RI, ARI, NMI)
    # =========================================================
    ri = rand_score(ground_truth_labels, final_labels)
    ari = adjusted_rand_score(ground_truth_labels, final_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, final_labels)

    unique_cameras = np.unique(final_labels)
    num_detected_clusters = len(unique_cameras)

    # =========================================================
    # 結果の表示 & ファイル出力
    # =========================================================
    output_dir = "./output_results"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 50)
    print(f"【最終評価結果】")
    print(f"  ・Ground Truth クラス数  : {len(unique_gt_names)} 個")
    print(f"  ・最終検出クラスタ数     : {num_detected_clusters} 個")
    print(f"  ・Rand Index (RI)        : {ri:.4f}")
    print(f"  ・Adjusted Rand Index(ARI): {ari:.4f}")
    print(f"  ・Normalized MI (NMI)    : {nmi:.4f}")
    print("=" * 50)

    print("\n[グループ分け詳細（先頭3枚の表示）]")
    for cam_id in unique_cameras:
        assigned_files = [
            os.path.basename(images_name[i])
            for i in range(n_samples)
            if final_labels[i] == cam_id
        ]
        print(
            f"  クラスタ {cam_id} ({len(assigned_files)}枚): {assigned_files[:3]}..."
        )

    # 1. CSVファイルへの保存
    csv_filename = os.path.join(output_dir, f"clustering_k{num_detected_clusters}_results.csv")
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "filename",
                "ground_truth_folder",
                "ground_truth_label",
                "predicted_cluster_id",
            ]
        )
        for path, gt_folder, gt_label, pred_label in zip(
            images_name,
            ground_truth_folders,
            ground_truth_labels,
            final_labels,
        ):
            writer.writerow(
                [
                    path,
                    os.path.basename(path),
                    gt_folder,
                    gt_label,
                    pred_label,
                ]
            )

    # 2. TXTサマリーレポートの保存
    txt_filename = os.path.join(output_dir, f"clustering_k{num_detected_clusters}_summary.txt")
    with open(txt_filename, mode="w", encoding="utf-8") as f:
        f.write("=== PRNU Clustering Results & Evaluation ===\n")
        f.write(f"Specified Target K     : {TARGET_K if TARGET_K else 'Auto'}\n")
        f.write(f"Ground Truth Classes   : {len(unique_gt_names)}\n")
        f.write(f"Predicted Clusters     : {num_detected_clusters}\n")
        f.write(f"Rand Index (RI)        : {ri:.4f}\n")
        f.write(f"Adjusted Rand Index(ARI): {ari:.4f}\n")
        f.write(f"Normalized MI (NMI)    : {nmi:.4f}\n")
        f.write("=" * 55 + "\n\n")

        for cam_id in unique_cameras:
            f.write(f"[推定クラスタ {cam_id}]\n")
            for i, path in enumerate(images_name):
                if final_labels[i] == cam_id:
                    f.write(
                        f"  - {os.path.basename(path)} (GT: {ground_truth_folders[i]})\n"
                    )

    print(f"\n結果を保存しました:")
    print(f"  ・CSV: {csv_filename}")
    print(f"  ・TXT: {txt_filename}")
    print("\n=== クラスタリング処理が正常に完了しました ===")


if __name__ == "__main__":
    main()