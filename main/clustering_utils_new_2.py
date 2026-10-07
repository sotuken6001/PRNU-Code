import io
import os
import gc
import zipfile
import cv2
import numpy as np
import bm3d
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm
from sklearn.metrics import normalized_mutual_info_score
import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------
# デバイス判定用（MPS / CUDA / CPU）
# ---------------------------------------------------------
_DEVICE = None

def _get_device():
    global _DEVICE
    if _DEVICE is None:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _DEVICE = torch.device('mps')
        elif torch.cuda.is_available():
            _DEVICE = torch.device('cuda')
        else:
            _DEVICE = torch.device('cpu')
    return _DEVICE

# =========================================================
# 前処理 & BM3D / Wavelet 関連
# ---------------------------------------------------------
def my_wavelet_wiener(X, sigma=5.0):
    """Haar ウェーブレットと局所 Wiener フィルタによるノイズ除去 (GPU高速処理)"""
    device = _get_device()
    x_tensor = torch.from_numpy(X).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    ll = torch.tensor([[0.5,  0.5], [ 0.5,  0.5]], device=device).view(1, 1, 2, 2)
    lh = torch.tensor([[-0.5, -0.5], [ 0.5,  0.5]], device=device).view(1, 1, 2, 2)
    hl = torch.tensor([[-0.5,  0.5], [-0.5,  0.5]], device=device).view(1, 1, 2, 2)
    hh = torch.tensor([[ 0.5, -0.5], [-0.5,  0.5]], device=device).view(1, 1, 2, 2)
    filters = torch.cat([ll, lh, hl, hh], dim=0)

    sigma2 = sigma ** 2
    coeffs = F.conv2d(x_tensor, filters, stride=2)
    ll_b, lh_b, hl_b, hh_b = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3], coeffs[:, 3:4]

    kernel_3x3 = torch.ones((1, 1, 3, 3), device=device) / 9.0
    filtered_subbands = [ll_b]

    for subband in [lh_b, hl_b, hh_b]:
        local_mean = F.conv2d(subband, kernel_3x3, padding=1)
        local_sq_mean = F.conv2d(subband ** 2, kernel_3x3, padding=1)
        local_var = torch.clamp(local_sq_mean - local_mean ** 2, min=0.0)

        signal_var = torch.clamp(local_var - sigma2, min=0.0)
        wiener_gain = signal_var / (signal_var + sigma2 + 1e-8)

        subband_filt = local_mean + wiener_gain * (subband - local_mean)
        filtered_subbands.append(subband_filt)

    coeffs_filt = torch.cat(filtered_subbands, dim=1)
    denoised_tensor = F.conv_transpose2d(coeffs_filt, filters, stride=2)
    return denoised_tensor.squeeze().cpu().numpy().astype(np.float32)


def get_crop_slice(img_shape, crop_size, crop_location):
    h, w = img_shape[:2]
    ch, cw = crop_size

    if ch == np.inf or cw == np.inf:
        return slice(None), slice(None)

    ch, cw = int(ch), int(cw)
    if h < ch or w < cw:
        return None, None

    if crop_location == 'center':
        y_start, x_start = (h - ch) // 2, (w - cw) // 2
    elif crop_location == 'upper-left':
        y_start, x_start = 0, 0
    else:
        raise ValueError("crop_location は 'center' か 'upper-left' を指定してください")

    return slice(y_start, y_start + ch), slice(x_start, x_start + cw)


def my_bm3d(X, sigma):
    X = X.astype(np.float32)
    max_val = np.max(X)
    is_0_255 = max_val > 10.0

    X_norm = X / 255.0 if is_0_255 else X
    sigma_norm = sigma / 255.0 if is_0_255 else sigma

    denoised = bm3d.bm3d(X_norm, sigma_norm, profile='np')
    if is_0_255:
        denoised *= 255.0

    return denoised.astype(np.float32)


def load_image_grayscale(img_path, zip_ref=None, zip_file_path=None):
    if zip_ref is not None or zip_file_path is not None:
        try:
            if zip_ref is not None:
                file_bytes = zip_ref.read(img_path)
            else:
                with zipfile.ZipFile(zip_file_path, 'r') as zf:
                    file_bytes = zf.read(img_path)
            
            img_array = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
            
            with Image.open(io.BytesIO(file_bytes)) as pil_img:
                return np.array(pil_img.convert('L'))
        except Exception:
            return None

    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img

    try:
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    except Exception:
        pass

    try:
        with Image.open(img_path) as pil_img:
            return np.array(pil_img.convert('L'))
    except Exception:
        return None


def extract_noise_residual(img_path, denoising_function, crop_size, crop_location, sigma=5, zip_ref=None, zip_file_path=None):
    img = load_image_grayscale(img_path, zip_ref=zip_ref, zip_file_path=zip_file_path)
    if img is None:
        return None

    sy, sx = get_crop_slice(img.shape, crop_size, crop_location)
    if sy is None:
        return None

    img_cropped = img[sy, sx]
    img_float = img_cropped.astype(np.float32)

    if denoising_function == 'wavelet':
        denoised_img = my_wavelet_wiener(img_float, sigma=sigma)
    elif denoising_function == 'bm3d':
        denoised_img = my_bm3d(img_float, sigma=sigma)
    elif denoising_function == 'none':
        denoised_img = img_float
    else:
        raise NotImplementedError(f"未対応のフィルター: {denoising_function}")

    residual = img_float - denoised_img
    residual = residual - np.mean(residual, dtype=np.float32)
    return residual


# =========================================================
# Step 1: 内積マトリクス計算
# =========================================================
def ResidualInnerProductMatrix(
    images_name,
    zip_file_path=None,
    denoising_function="wavelet",
    crop_size=(500, 500),
    crop_location="center",
    device=None,
    batch_size=1000,
):
    if device is None:
        device = _get_device()

    print(f"使用デバイス: {device}")

    num_images = len(images_name)
    feature_dim = crop_size[0] * crop_size[1]

    residuals_matrix = np.zeros((num_images, feature_dim), dtype=np.float16)

    valid_paths = []
    skipped_paths = []
    valid_count = 0

    print("Step 1/2: 各画像からノイズ（指紋）を抽出中...")

    def process_and_store(path, z_ref=None):
        nonlocal valid_count
        res = extract_noise_residual(
            path,
            denoising_function,
            crop_size,
            crop_location,
            zip_ref=z_ref,
            zip_file_path=zip_file_path if z_ref is None else None,
        )

        if res is None:
            skipped_paths.append(path)
            return

        res_flat = res.astype(np.float32).ravel()
        norm_val = np.linalg.norm(res_flat)
        if norm_val > 0:
            res_normalized = res_flat / norm_val
        else:
            res_normalized = np.zeros_like(res_flat)

        residuals_matrix[valid_count] = res_normalized.astype(np.float16)
        valid_paths.append(path)
        valid_count += 1

    if zip_file_path is not None:
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            for path in tqdm(images_name):
                process_and_store(path, z_ref=zip_ref)
    else:
        for path in tqdm(images_name):
            process_and_store(path)

    print("\n" + "=" * 50)
    print("【前処理（Step 1/2）画像読み込み結果】")
    print(f"  ・対象の総画像数      : {num_images:,} 枚")
    print(f"  ・正常に読み込めた画像: {valid_count:,} 枚")
    print(f"  ・スキップされた画像  : {skipped_count:,} 枚" if (skipped_count := len(skipped_paths)) else "  ・スキップ画像なし")
    print("=" * 50 + "\n")

    if valid_count == 0:
        raise ValueError("有効な画像が1枚も読み込めませんでした。")

    residuals_matrix = residuals_matrix[:valid_count]
    images_name[:] = valid_paths
    gc.collect()

    print(f"Step 2/2: GPU分割行列積（内積マトリクス {valid_count}x{valid_count}）を計算中...")

    C = np.zeros((valid_count, valid_count), dtype=np.float32)

    for i in range(0, valid_count, batch_size):
        end_i = min(i + batch_size, valid_count)
        X_i = torch.from_numpy(residuals_matrix[i:end_i].astype(np.float32)).to(device)

        for j in range(i, valid_count, batch_size):
            end_j = min(j + batch_size, valid_count)
            X_j = torch.from_numpy(residuals_matrix[j:end_j].astype(np.float32)).to(device)

            sub_C = torch.mm(X_i, X_j.T).cpu().numpy()
            C[i:end_i, j:end_j] = sub_C
            if i != j:
                C[j:end_j, i:end_i] = sub_C.T

        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return C


# =========================================================
# Step 2: ベースクラスタリング生成 (Correlation Clustering with Sweep alpha)
# =========================================================
def estimate_sigma0(rho_matrix):
    """クロスカメラ相関 (H0) の標準偏差 sigma_0 の推定"""
    triu_indices = np.triu_indices_from(rho_matrix, k=1)
    corrs = rho_matrix[triu_indices]
    # 相関分布の大半は H0 (クロスカメラ) 由来であるため、堅牢な標準偏差(MAD由来)で推定
    median = np.median(corrs)
    mad = np.median(np.abs(corrs - median))
    sigma0 = 1.4826 * mad
    return max(sigma0, 1e-4)


def greedy_correlation_clustering(rho_matrix, alpha):
    """単一の alpha に対する Greed/Pivoting 相関クラスタリング"""
    n = rho_matrix.shape[0]
    W = rho_matrix - alpha
    
    unassigned = set(range(n))
    labels = np.zeros(n, dtype=int)
    cluster_id = 0

    while unassigned:
        # 正のエッジ和が最大のノードをピボットに選択
        pivot = max(unassigned, key=lambda i: np.sum(W[i, list(unassigned)]))
        cluster = {pivot}
        unassigned.remove(pivot)

        # ピボット集合との正の関連度を持つノードを探索して追加
        candidates = list(unassigned)
        for cand in candidates:
            if np.mean([W[cand, member] for member in cluster]) > 0:
                cluster.add(cand)
                unassigned.remove(cand)

        for member in cluster:
            labels[member] = cluster_id
        cluster_id += 1

    return labels


def generate_base_clusterings(C, num_alphas=41):
    """
    論文 Section IV-B:
    alpha を [1.0*sigma_0, 5.0*sigma_0] の範囲でスウィープして複数のベースクラスタリングを生成
    """
    # 内積 C から相関行列 rho を導出
    diag = np.diag(C)
    inv_norm = 1.0 / np.sqrt(np.maximum(diag, 1e-8))
    rho_matrix = C * np.outer(inv_norm, inv_norm)
    np.fill_diagonal(rho_matrix, 1.0)

    sigma0 = estimate_sigma0(rho_matrix)
    alphas = np.linspace(1.0 * sigma0, 5.0 * sigma0, num_alphas)

    print(f"Correlation Clustering alpha スウィープ実行中 (sigma0 = {sigma0:.5f}, alpha 範囲: [{alphas[0]:.5f}, {alphas[-1]:.5f}])...")

    base_partitions = []
    for alpha in alphas:
        labels = greedy_correlation_clustering(rho_matrix, alpha)
        base_partitions.append(labels)

    return np.array(base_partitions)  # Shape: (M, N)


# =========================================================
# Step 3: 合意クラスタリング (WEAC & 自動 l* 選択)
# =========================================================

def weac_consensus_clustering(base_partitions, target_k=None, epsilon=0.01):
    """
    Weighted Evidence Accumulation Clustering (WEAC)
    - target_k が指定されている場合: そのクラスタ数 k の分割を返す
    - target_k が None の場合: 相似度変化 s(l) により最適 l* を自動判定する
    """
    M, N = base_partitions.shape
    if M == 1:
        return base_partitions[0]

    # 1. 各ベースクラスタリング間の NMI を計算
    sim_matrix = np.zeros((M, M))
    for i in range(M):
        for j in range(i, M):
            score = normalized_mutual_info_score(base_partitions[i], base_partitions[j])
            sim_matrix[i, j] = score
            sim_matrix[j, i] = score

    # 2. Crowd Agreement Index (CAI) & 重み w_i
    cai = np.sum(sim_matrix, axis=1) - np.diag(sim_matrix)
    cai = cai / (M - 1 + 1e-8)
    
    weights = cai ** 2
    weight_sum = np.sum(weights)
    if weight_sum > 0:
        weights /= weight_sum
    else:
        weights = np.ones(M) / M

    # 3. 加重共出現行列 (Co-association Matrix A)
    A = np.zeros((N, N), dtype=np.float32)
    for k in range(M):
        labels = base_partitions[k]
        match = (labels[:, None] == labels[None, :])
        A += match.astype(np.float32) * weights[k]

    # 4. 距離行列 D = 1 - A による Hierarchical Clustering (Average Linkage)
    dist_matrix = np.clip(1.0 - A, 0.0, 1.0)
    dist_vector = squareform(dist_matrix, checks=False)
    Z = linkage(dist_vector, method='average')

    # ★ target_k が指定されている場合はそのクラスタ数を直接切り出して返す
    if target_k is not None:
        target_k = min(target_k, N)
        labels = fcluster(Z, t=target_k, criterion='maxclust')
        print(f"WEAC 合意クラスタリング完了: 指定クラスタ数 k = {target_k}")
        return labels

    # 5. target_k が None の場合は自動判定: s(l) = NMI(Q_{l-1}, Q_l)
    max_l = min(N, 100)
    partitions_by_l = {}
    for l in range(1, max_l + 1):
        partitions_by_l[l] = fcluster(Z, t=l, criterion='maxclust')

    s_vals = {}
    for l in range(2, max_l + 1):
        s_vals[l] = normalized_mutual_info_score(partitions_by_l[l - 1], partitions_by_l[l])

    opt_l = max_l
    for l in range(2, max_l):
        if s_vals[l] < 1.0 - epsilon:
            if all(s_vals[k] >= 1.0 - epsilon for k in range(l + 1, max_l + 1)):
                opt_l = l
                break

    print(f"WEAC 合意クラスタリング完了: 最適クラスタ数 l* = {opt_l}")
    return partitions_by_l[opt_l]

# =========================================================
# Step 4: クラスタ精錬 (Refinement Step via Lambda_pq)
# =========================================================
def _calc_cluster_corr(C, cluster_indices, target_indices):
    """
    論文の式 (33): 内積行列 C から PRNU 推定値 K_p と各残差 R_i の相関 corr(K_p, R_i) を高速計算
    """
    if len(cluster_indices) == 0 or len(target_indices) == 0:
        return np.array([])

    sum_c_j_i = np.sum(C[cluster_indices][:, target_indices], axis=0)  # Shape: (len(target_indices),)
    sum_c_jk = np.sum(C[cluster_indices][:, cluster_indices])
    c_ii = np.diag(C)[target_indices]

    denom = np.sqrt(np.maximum(sum_c_jk, 1e-8)) * np.sqrt(np.maximum(c_ii, 1e-8))
    corr = sum_c_j_i / np.maximum(denom, 1e-8)
    return np.clip(corr, -1.0, 1.0)


def _calc_loo_corr(C, cluster_indices):
    """
    Leave-One-Out (LOO) 相関の計算: K_{p,\j} と R_j の相関 (式 31-32)
    """
    cp_size = len(cluster_indices)
    if cp_size <= 1:
        return np.array([0.0])

    c_sub = C[cluster_indices][:, cluster_indices]
    sum_c_jk = np.sum(c_sub)
    col_sums = np.sum(c_sub, axis=1)
    c_jj = np.diag(c_sub)

    # LOO 分子 & 分母
    numer = col_sums - c_jj
    denom_sq = sum_c_jk - 2.0 * col_sums + c_jj
    denom = np.sqrt(np.maximum(denom_sq, 1e-8)) * np.sqrt(np.maximum(c_jj, 1e-8))

    loo_corrs = numer / np.maximum(denom, 1e-8)
    return np.clip(loo_corrs, -1.0, 1.0)


def refine_clustering_paper(C, initial_labels):
    """
    論文 Section IV-D & Algorithm 1 に完全準拠した対数尤度比 Lambda_pq による精錬ステップ
    """
    n_samples = C.shape[0]
    labels = initial_labels.copy()

    # ユニークなクラスタ ID 管理
    unique_ids = np.unique(labels)
    clusters = {cid: np.where(labels == cid)[0] for cid in unique_ids}

    T_C = 0  # 実行時閾値

    print("Step 3/3: 対数尤度比統計量 (Lambda_pq) による Refinement を実行中...")

    while True:
        T_C += 1
        changed_in_pass = False

        while True:
            # 大クラスタ (L) と小クラスタ (S) の分離
            L_keys = [k for k, v in clusters.items() if len(v) > T_C]
            S_keys = [k for k, v in clusters.items() if len(v) <= T_C]

            if not L_keys or not S_keys:
                break

            best_lambda = -np.inf
            best_pair = None

            # 各大クラスタ C_p について H0, H1 分布をパラメータ推定
            for p_key in L_keys:
                p_idx = clusters[p_key]

                # H1 分布: LOO 相関から推定
                loo_corrs = _calc_loo_corr(C, p_idx)
                mu1, std1 = np.mean(loo_corrs), np.std(loo_corrs)
                std1 = max(std1, 1e-3)

                # 他の全要素との相関
                out_idx = np.setdiff1d(np.arange(n_samples), p_idx)
                if len(out_idx) == 0:
                    continue

                corrs_out = _calc_cluster_corr(C, p_idx, out_idx)
                
                # H0 分布: クラスタ外要素の相関から推定
                mu0, std0 = np.mean(corrs_out), np.std(corrs_out)
                std0 = max(std0, 1e-3)

                # 各小クラスタ C_q に対する対数尤度比 Lambda_pq の計算
                for q_key in S_keys:
                    if p_key == q_key:
                        continue

                    q_idx = clusters[q_key]
                    rho_q = _calc_cluster_corr(C, p_idx, q_idx)

                    # 対数尤度 L(R_i | H1) - L(R_i | H0)
                    log_pdf1 = norm.logpdf(rho_q, loc=mu1, scale=std1)
                    log_pdf0 = norm.logpdf(rho_q, loc=mu0, scale=std0)

                    lambda_pq = np.sum(log_pdf1 - log_pdf0)

                    if lambda_pq > best_lambda:
                        best_lambda = lambda_pq
                        best_pair = (p_key, q_key)

            # Lambda_pq > 0 である場合のみ統合
            if best_lambda > 0 and best_pair is not None:
                p_win, q_win = best_pair
                clusters[p_win] = np.concatenate([clusters[p_win], clusters[q_win]])
                del clusters[q_win]
                changed_in_pass = True
            else:
                break

        # 大クラスタが1つのみ、または統合が発生しなくなったら終了
        remaining_large = [k for k, v in clusters.items() if len(v) > T_C]
        if len(remaining_large) <= 1 and not changed_in_pass:
            break

        # 安全ガード（クラスタ数が変化しなくなったら終了）
        if not changed_in_pass and T_C > max([len(v) for v in clusters.values()]):
            break

    # 最終ラベル付け
    final_labels = np.zeros(n_samples, dtype=int)
    for new_id, (k, idxs) in enumerate(clusters.items()):
        final_labels[idxs] = new_id

    return final_labels


# =========================================================
# エントリーポイント用パイプライン関数
# =========================================================
def run_blind_prnu_clustering(images_name, zip_file_path=None, denoising_function="wavelet"):
    """
    論文の手法全体を実行するパイプライン関数
    """
    # 1. 内積マトリクス C の作成
    C = ResidualInnerProductMatrix(
        images_name=images_name,
        zip_file_path=zip_file_path,
        denoising_function=denoising_function,
        crop_size=(500, 500),
        crop_location="center"
    )

    # 2. ベースクラスタリング群の生成 (alpha スウィープ)
    base_partitions = generate_base_clusterings(C)

    # 3. WEAC による合意クラスタリング (保守的初期クラスタ Q* の生成)
    initial_consensus_labels = weac_consensus_clustering(base_partitions)

    # 4. 対数尤度比 (Lambda_pq) による精錬 (Refinement)
    final_labels = refine_clustering_paper(C, initial_consensus_labels)

    print("\n" + "=" * 50)
    print(f"【ブラインドクラスタリング最終結果】")
    print(f"  ・検出された推定デバイス（クラスタ）数: {len(np.unique(final_labels))} 個")
    print("=" * 50 + "\n")

    return final_labels