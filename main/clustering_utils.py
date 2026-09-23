# # 内積計算
# import numpy as np
# import bm3d

# def scipy_warning(message):
#     import sys
#     print(f"Warning: {message}", file=sys.stderr)

# def my_bm3d(X, sigma):
#     """
#     MATLABの bm3d.m の挙動を再現する関数。
#     入力X: NumPy配列 (H, W) または (H, W, C)、範囲は任意 (0-255等)
#     """
#     # 入力チェック
#     if not isinstance(sigma, (int, float)):
#         raise ValueError('"sigma" はスカラ値でなければなりません')

#     # BM3Dライブラリは0〜1（あるいは0〜255）のfloat型、かつ2Dまたは3D配列を受け付けます
#     # MATLAB版は内部で各チャンネルを独立して `actual_BM3D` にかけ、0-255の範囲にスケーリングしています
#     X = np.atleast_3d(X).astype(np.float64)
#     Nr, Nc, Nb = X.shape
#     Y = np.zeros_like(X)

#     # 低複雑度プロファイル 'lc' に相当する設定として、
#     # bm3dライブラリの stage_arg（BM3DProfile）を調整することも可能ですが、
#     # 通常はライブラリの標準最適化を利用します。
#     # ここではMATLAB版同様、チャンネルごとにループ処理します。
#     for i in range(Nb):
#         channel = X[:, :, i]

#         # 内部の actual_BM3D は [0, 1] への正規化を考慮しているため、
#         # 入力が 0-255 の場合は、シグマを 255 で割って正規化空間で処理します
#         max_val = np.max(channel)
#         is_0_255 = max_val > 10  # MATLAB版の naive check (max(z(:)) > 10) を再現

#         if is_0_255:
#             channel_norm = channel / 255.0
#             sigma_norm = sigma / 255.0
#         else:
#             channel_norm = channel
#             sigma_norm = sigma

#         # BM3Dによるノイズ除去を実行
#         # bm3d.BM3D_PROFILE_FAST は MATLAB の 'lc' (Low Complexity) に近いです
#         denoised = bm3d.bm3d(channel_norm, sigma_norm, profile='np')
        
#         if is_0_255:
#             Y[:, :, i] = denoised * 255.0
#         else:
#             Y[:, :, i] = denoised

#     # 入力がもともと2次元（グレースケール）なら、2次元に絞って返す
#     if Y.shape[2] == 1:
#         Y = Y.squeeze(axis=2)

#     return Y


# def my_cbm3d(X, sigma):
#     """
#     MATLABの cbm3d.m の挙動を再現する関数。
#     カラー画像の場合、輝度・色差空間（YUV / Opponent空間）に変換してノイズ除去を行います。
#     """
#     if not isinstance(sigma, (int, float)):
#         raise ValueError('"sigma" はスカラ値でなければなりません')

#     X = np.array(X, dtype=np.float64)

#     if X.ndim == 3 and X.shape[2] == 3:
#         # カラー画像の場合
#         # MATLAB版の naive check を再現し、必要に応じて0-1に正規化
#         max_val = np.max(X)
#         is_0_255 = max_val > 10

#         if is_0_255:
#             X_norm = X / 255.0
#             sigma_norm = sigma / 255.0
#         else:
#             X_norm = X
#             sigma_norm = sigma

#         # Pythonのbm3dライブラリには、カラー対応の `bm3d_rgb` が用意されており、
#         # 内部で自動的に輝度・色差（Opponent）空間への変換と適切なノイズ除去を行ってくれます。
#         denoised = bm3d.bm3d_rgb(X_norm, sigma_norm, profile='np')

#         if is_0_255:
#             return denoised * 255.0
#         else:
#             return denoised
#     else:
#         # グレースケール画像の場合は通常のbm3dを呼び出す
#         return my_bm3d(X, sigma)


# def bm3d_log(X, sigma):
#     """
#     MATLABの bm3d_log.m の挙動を完全再現する関数。
#     """
#     X = np.array(X, dtype=np.float64)
#     eps = np.finfo(float).eps  # MATLABの `eps` に相当する微小値

#     # 1. 対数変換（ドメイン変換）
#     Xlog = np.log(X + eps)

#     # 2. ノイズ平均の減算（MATLABコード内で mean_val = 0 なので実質変化なし）
#     mean_val = 0
#     Xlog = Xlog - mean_val

#     # 3. 0-255 の範囲へスケーリング
#     mi = np.min(Xlog)
#     Mi = np.max(Xlog)

#     # 万が一、分母が0になる（画像が完全に単色）場合の対策
#     if Mi - mi == 0:
#         Xlog_scaled = np.zeros_like(Xlog)
#     else:
#         Xlog_scaled = (255.0 / (Mi - mi)) * (Xlog - mi)

#     # 4. ノイズ除去（上で定義した my_bm3d を呼び出す）
#     Ylog = my_bm3d(Xlog_scaled, sigma)

#     # MATLABコードの `Ylog = 255 * Ylog;` は、実際の実際上のバグ、
#     # もしくは特殊な入力（[0,1]で戻ってきた場合）の補正処理です。
#     # Pythonのbm3dは入力スケールを維持して返すため、
#     # MATLABの `255 * Ylog` 自体はスケーリングを元に戻すステップ5と相殺する形で調整します。

#     # 5. 元の範囲に逆スケーリング
#     if Mi - mi != 0:
#         Ylog = Ylog / (255.0 / (Mi - mi)) + mi

#     # 6. 線形ドメインに逆変換（指数変換）
#     Y = np.exp(Ylog) - eps

#     return Y

# # --- 使用例 ---
# if __name__ == "__main__":
#     # テスト用のダミー画像（128x128のカラー画像）を作成
#     np.random.seed(0)
#     img_pure = np.ones((128, 128, 3)) * 128
#     noise = np.random.normal(0, 15, img_pure.shape)
#     img_noisy = np.clip(img_pure + noise, 0, 255)

#     print("元のノイズ画像形状:", img_noisy.shape)

#     # 1. 通常のBM3D
#     out_bm3d = my_bm3d(img_noisy, sigma=15)
#     print("my_bm3d 完了")

#     # 2. カラー用BM3D
#     out_cbm3d = my_cbm3d(img_noisy, sigma=15)
#     print("my_cbm3d 完了")

#     # 3. 対数ドメインBM3D
#     out_log = bm3d_log(img_noisy, sigma=15)
#     print("bm3d_log 完了")

# # 比較表の作成

# import numpy as np
# import cv2
# import os
# from tqdm import tqdm
# # 先ほど作成した bm3d のラッパー関数をインポート（または同じファイルに記述）
# # from bm3d_utils import my_bm3d

# def get_crop_slice(img_shape, crop_size, crop_location):
#     """
#     画像の切り抜き（Crop）範囲を計算する補助関数
#     """
#     h, w = img_shape[:2]
#     ch, cw = crop_size

#     # サイズが Inf（無限大）に指定されている場合は切り抜かない
#     if ch == np.inf or cw == np.inf:
#         return slice(None), slice(None)

#     ch, cw = int(ch), int(cw)

#     if crop_location == 'center':
#         y_start = (h - ch) // 2
#         x_start = (w - cw) // 2
#     elif crop_location == 'upper-left':
#         y_start, x_start = 0, 0
#     else:
#         raise ValueError("crop_location は 'center' か 'upper-left' を指定してください")

#     return slice(y_start, y_start + ch), slice(x_start, x_start + cw)

# def extract_noise_residual(img_path, denoising_function, crop_size, crop_location, sigma=5):
#     """
#     1枚の画像からノイズ（残差 = 元画像 - 除去後画像）を抽出する補助関数
#     """
#     if not os.path.exists(img_path):
#         raise FileNotFoundError(f"画像が見つかりません: {img_path}")

#     # PRNU（カメラ指紋）の抽出は一般的にグレースケールで行われます
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"画像の読み込みに失敗しました: {img_path}")

#     img = img.astype(np.float64)

#     # 1. フィルターで「被写体の景色」成分を作る
#     if denoising_function == 'bm3d':
#         # ※ここで先ほど作った my_bm3d を呼び出します
#         denoised_img = my_bm3d(img, sigma=sigma)
#     elif denoising_function == 'none':
#         denoised_img = img
#     else:
#         raise NotImplementedError(f"未対応のフィルター: {denoising_function}")

#     # 2. 残差（ノイズ）の計算: ノイズ = 元画像 - 景色
#     residual = img - denoised_img

#     # 3. 指定サイズに切り抜き
#     sy, sx = get_crop_slice(residual.shape, crop_size, crop_location)
#     residual_cropped = residual[sy, sx]

#     # 4. ノイズの平均をゼロにする（比較計算の精度を上げるための一般的な前処理）
#     residual_cropped = residual_cropped - np.mean(residual_cropped)

#     return residual_cropped

# def ResidualInnerProductMatrix(images_name, denoising_function='bm3d', crop_size=[np.inf, np.inf], crop_location='center', in_memory=True):
#     """
#     総当たりの内積マトリクス（C）を計算するメイン関数
#     """
#     num_images = len(images_name)
#     C = np.eye(num_images)  # 対角成分が1（自分自身との比較）の単位行列を作成

#     residuals = []

#     print("Step 1/2: 各画像からノイズ（指紋）を抽出中...")
#     for path in tqdm(images_name):
#         res = extract_noise_residual(path, denoising_function, crop_size, crop_location)

#         # 内積計算を高速化するため、あらかじめベクトル長を1に正規化（単位ベクトル化）しておく
#         norm = np.linalg.norm(res)
#         if norm > 0:
#             res_normalized = res / norm
#         else:
#             res_normalized = np.zeros_like(res)

#         residuals.append(res_normalized)

#     print("Step 2/2: 総当たりで類似度（内積マトリクス）を計算中...")
#     # 総当たり戦（組み合わせ）の計算
#     for i in tqdm(range(num_images)):
#         for j in range(i + 1, num_images):
#             # 2つのノイズ成分を重ね合わせて、どれくらい一致するか（内積）を計算
#             # np.vdot は平坦化して内積をとるため、2次元の画像マトリクス同士の比較に最適
#             similarity = np.vdot(residuals[i], residuals[j])

#             # Cマトリクスは対称行列（AとBの類似度 ＝ BとAの類似度）なので両方に代入
#             C[i, j] = similarity
#             C[j, i] = similarity

#     return C

# # エネルギー関数の計算

# # import numpy as np
# # import scipy.sparse as sp

# # # ※注意: 実際に動かすには QPBO のPythonラッパーライブラリが必要です。
# # # 例: pip install pyqpbo (環境によってはC++コンパイラの設定が必要になります)
# # try:
# #     import pyqpbo
# # except ImportError:
# #     print("Warning: pyqpbo がインストールされていません。ダミー関数で代用します。")

# # def CCEnergy(w, l):
# #     """
# #     相関クラスタリングのエネルギーを計算する関数（CCEnergy.m に相当）
# #     エネルギーが低い（マイナスに大きい）ほど、綺麗にグループ分けできている証拠です。

# #     Parameters:
# #         w (scipy.sparse.csr_matrix): 類似度のスパース行列（対角成分は0）
# #         l (numpy.ndarray): 各画像の現在のラベル（グループID）
# #     """
# #     # ゼロ以外の要素（線が引かれている画像ペア）のインデックスを取得
# #     ii, jj = w.nonzero()

# #     # 同じグループに属しているペアだけを抽出
# #     same_label = l[ii] == l[jj]

# #     # 同じグループ同士の線の太さ（重み）の合計をマイナスにする
# #     # （仲良しが同じグループにいるほどエネルギーが下がる）
# #     energy = -np.sum(w.data[same_label])

# #     return energy / 2.0  # 対称行列で2重カウントされるため半分にする

# # def BinaryExpand(w, l, li):
# #     """
# #     特定のグループ(li)の陣地を広げる最適化処理（BinaryExpand.m に相当）
# #     """
# #     nl = l.copy()

# #     # すでにラベル「li」を持っているノードは計算から除外
# #     non_li_idx = np.where(l != li)[0]
# #     n = len(non_li_idx)

# #     if n == 0:
# #         return nl  # すべてのノードがすでに「li」なら何もしない

# #     # --- 1. Unary項（1つのノード単体に対するコスト）の計算 ---
# #     # 0: 今のラベルを維持するコスト
# #     # 1: 新しくラベル「li」に乗り換えるコスト
# #     unary = np.zeros((n, 2), dtype=np.float64)

# #     # ノードが「li」に乗り換えた場合、既存の「li」ノードとのつながり（重み）が得られる
# #     is_li = (l == li)
# #     w_non_li = w[non_li_idx, :]
# #     # scipy.sparseの行列積を使って高速に計算
# #     unary[:, 1] = -np.array(w_non_li[:, is_li].sum(axis=1)).flatten()

# #     # --- 2. Pairwise項（ノード同士のつながりに対するコスト）の計算 ---
# #     # 対象ノード同士の部分行列を抽出（下三角行列のみ）
# #     w_sub = sp.tril(w[non_li_idx, :][:, non_li_idx], -1)
# #     ii, jj, wij = sp.find(w_sub)

# #     edges = np.vstack((ii, jj)).T.astype(np.int32)
# #     edge_weights = np.zeros((len(ii), 4), dtype=np.float64)

# #     rl = l[non_li_idx]
# #     old_differ = (rl[ii] != rl[jj]).astype(np.float64)

# #     # QPBOに渡すための4つの状態コスト (E00, E01, E10, E11)
# #     # E00: 両方とも今のラベルを維持
# #     # E01 / E10: 片方だけが「li」に乗り換える
# #     # E11: 両方とも「li」に乗り換える
# #     edge_weights[:, 0] = wij * old_differ  # E00
# #     edge_weights[:, 1] = wij               # E01
# #     edge_weights[:, 2] = wij               # E10
# #     edge_weights[:, 3] = 0.0               # E11 (両方liになるのでコスト0)

# #     # --- 3. QPBOアルゴリズムによる最適化 ---
# #     if 'pyqpbo' in globals():
# #         # pyqpbo を使ってグラフカットを解く (0: 維持, 1: 乗り換え)
# #         result_labels = pyqpbo.solve_qpbo(unary, edges, edge_weights)
# #         changed = (result_labels == 1)
# #         nl[non_li_idx[changed]] = li
# #     else:
# #         # ※ライブラリがない場合のダミー処理（実際にはここにQPBOが入ります）
# #         pass

# #     return nl

# # def a_expand(w, ig=None):
# #     """
# #     Alpha-Expansionのメインループ（a_expand.m に相当）
# #     """
# #     n = w.shape[0]

# #     # 対角成分（自分自身との類似度）を0にする
# #     w = w.copy()
# #     w.setdiag(0)
# #     w.eliminate_zeros()

# #     # 初期ラベル（ig）の準備
# #     if ig is not None and len(ig) == n:
# #         # 重複を省いて連番に変換（MATLABの unique 相当）
# #         _, l = np.unique(ig, return_inverse=True)
# #     else:
# #         l = np.ones(n, dtype=np.int32)

# #     NL = len(np.unique(l))
# #     current_energy = CCEnergy(w, l)

# #     while True:
# #         accepted = False
# #         li = 0

# #         while li < NL:
# #             # 各グループ（li）について、陣地を広げられるかテストする
# #             nl = BinaryExpand(w, l, li)
# #             new_energy = CCEnergy(w, nl)

# #             # エネルギーが下がった（より綺麗にグループ分けできた）場合、採用！
# #             if new_energy < current_energy:
# #                 current_energy = new_energy
# #                 l = nl
# #                 accepted = True
# #                 NL = np.max(nl) + 1  # 最大ラベルIDを更新

# #             li += 1

# #         # どのグループを広げようとしても改善しなくなったらループ終了
# #         if not accepted:
# #             break

# #         # 空になったグループを詰めて整理する
# #         _, l = np.unique(l, return_inverse=True)
# #         NL = len(np.unique(l))

# #     return l

# # エネルギー
# import numpy as np

# def al_icm(w, initial_labels=None, limit=100):
#     """
#     ICM (Iterative Conditional Modes) に基づくグラフエネルギー最適化アルゴリズム。
#     アフィニティ行列 W（類似度から閾値を引いた正負の値を持つ行列）に基づき、
#     各要素のクラスタ割り当てを反復的に更新します。
    
#     Parameters:
#     -----------
#     w : numpy.ndarray
#         N x N のアフィニティ/類似度行列 (対角成分以外に正負の値を含む)
#     initial_labels : numpy.ndarray, optional
#         初期クラスタラベル。None の場合は全要素を同じ初期ラベルに設定
#     limit : int, default=100
#         最大反復回数
        
#     Returns:
#     --------
#     labels : numpy.ndarray
#         最適化後のクラスタラベル (0, 1, 2... に再採番された配列)
#     """
#     n = w.shape[0]
#     w_clean = w.copy()
#     np.fill_diagonal(w_clean, 0)
    
#     if initial_labels is None or len(initial_labels) != n:
#         labels = np.ones(n, dtype=int)
#     else:
#         _, labels = np.unique(initial_labels, return_inverse=True)
        
#     for itr in range(limit):
#         prev_labels = labels.copy()
#         unique_labels = np.unique(labels)
#         max_label = unique_labels.max()
        
#         for i in range(n):
#             best_label = labels[i]
#             max_affinity = -np.inf
            
#             # 既存の全クラスタに加え、新規クラスタ生成の候補（max_label + 1）を追加
#             candidate_labels = list(unique_labels)
#             candidate_labels.append(max_label + 1)
            
#             for c in candidate_labels:
#                 # 対象要素 i と クラスタ c に属する他要素との親和性（アフィニティ）の総和を計算
#                 affinity = np.sum(w_clean[i, labels == c])
#                 if affinity > max_affinity:
#                     max_affinity = affinity
#                     best_label = c
                    
#             labels[i] = best_label
#             if best_label > max_label:
#                 unique_labels = np.unique(labels)
#                 max_label = unique_labels.max()

#         # 15回ごとにラベルの番号詰め（0からの連番へ再割当）を実施
#         if itr % 15 == 0:
#             _, labels = np.unique(labels, return_inverse=True)
            
#         # 変化がなくなったら早期終了
#         if np.array_equal(prev_labels, labels):
#             break
            
#     # 最終的なラベルを 0 からの連番にリセットして返却
#     _, labels = np.unique(labels, return_inverse=True)
#     return labels

# # WEACフェーズ

# import numpy as np
# from scipy.cluster.hierarchy import linkage, fcluster
# from scipy.spatial.distance import squareform

# def weac_python(base_cls, ncai, cls_nums):
#     """
#     WEACアルゴリズムのPython実装
#     base_cls: (N x M) 各列がクラスタリング結果
#     ncai: (M,) 各クラスタリングの重み
#     cls_nums: 分割したいクラスタ数のリスト
#     """
#     n_samples, n_base = base_cls.shape
#     ncai = ncai.flatten()
#     weight_sum = np.sum(ncai)

#     # 1. 重み付き類似度行列の構築 (Co-association matrix)
#     S = np.zeros((n_samples, n_samples))
#     for k in range(n_base):
#         # 各サンプルが同じクラスタに属しているかどうかの行列を作成
#         labels = base_cls[:, k]
#         # ブロードキャストで各ペアが同じクラスタか判定
#         match = (labels[:, None] == labels[None, :])
#         S += match.astype(float) * ncai[k]

#     S = S / weight_sum

#     # 2. 距離行列への変換 (similarity to distance: 1 - S)
#     # scipyのlinkage関数は距離ベクトルを必要とするため
#     dist_matrix = 1 - S
#     dist_vector = squareform(dist_matrix, checks=False)

#     # 3. 階層的クラスタリングの実行 (Average Linkageを使用)
#     Z = linkage(dist_vector, method='average')

#     # 4. 指定されたクラスタ数での分割
#     results = {}
#     for k in cls_nums:
#         results[k] = fcluster(Z, t=k, criterion='maxclust')

#     return results

# # 使用例:
# # ncai = getNCAI_python(base_cls) # ※getNCAIも同様に移植が必要
# # consensus_results = weac_python(base_cls, ncai, [2, 3, 4])

# 改良版
import os
import cv2
import numpy as np
import bm3d
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# =========================================================
# 前処理 & BM3D 関連
# =========================================================
def my_bm3d(X, sigma):
    X = np.atleast_3d(X).astype(np.float64)
    Nr, Nc, Nb = X.shape
    Y = np.zeros_like(X)

    for i in range(Nb):
        channel = X[:, :, i]
        max_val = np.max(channel)
        is_0_255 = max_val > 10

        if is_0_255:
            channel_norm = channel / 255.0
            sigma_norm = sigma / 255.0
        else:
            channel_norm = channel
            sigma_norm = sigma

        denoised = bm3d.bm3d(channel_norm, sigma_norm, profile='np')

        if is_0_255:
            Y[:, :, i] = denoised * 255.0
        else:
            Y[:, :, i] = denoised

    if Y.shape[2] == 1:
        Y = Y.squeeze(axis=2)

    return Y

def my_cbm3d(X, sigma):
    X = np.array(X, dtype=np.float64)
    if X.ndim == 3 and X.shape[2] == 3:
        max_val = np.max(X)
        is_0_255 = max_val > 10

        if is_0_255:
            X_norm = X / 255.0
            sigma_norm = sigma / 255.0
        else:
            X_norm = X
            sigma_norm = sigma

        denoised = bm3d.bm3d_rgb(X_norm, sigma_norm, profile='np')

        if is_0_255:
            return denoised * 255.0
        else:
            return denoised
    else:
        return my_bm3d(X, sigma)

def get_crop_slice(img_shape, crop_size, crop_location):
    h, w = img_shape[:2]
    ch, cw = crop_size

    if ch == np.inf or cw == np.inf:
        return slice(None), slice(None)

    ch, cw = int(ch), int(cw)

    if crop_location == 'center':
        y_start = (h - ch) // 2
        x_start = (w - cw) // 2
    elif crop_location == 'upper-left':
        y_start, x_start = 0, 0
    else:
        raise ValueError("crop_location は 'center' か 'upper-left' を指定してください")

    return slice(y_start, y_start + ch), slice(x_start, x_start + cw)

def extract_noise_residual(img_path, denoising_function, crop_size, crop_location, sigma=5):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"画像が見つかりません: {img_path}")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"画像の読み込みに失敗しました: {img_path}")

    img = img.astype(np.float64)

    if denoising_function == 'bm3d':
        denoised_img = my_bm3d(img, sigma=sigma)
    elif denoising_function == 'none':
        denoised_img = img
    else:
        raise NotImplementedError(f"未対応のフィルター: {denoising_function}")

    residual = img - denoised_img

    sy, sx = get_crop_slice(residual.shape, crop_size, crop_location)
    residual_cropped = residual[sy, sx]
    residual_cropped = residual_cropped - np.mean(residual_cropped)

    return residual_cropped

# =========================================================
# Step 1: 内積マトリクス計算
# =========================================================
def ResidualInnerProductMatrix(images_name, denoising_function='bm3d', crop_size=[np.inf, np.inf], crop_location='center'):
    num_images = len(images_name)
    C = np.eye(num_images)

    residuals = []

    print("Step 1/2: 各画像からノイズ（指紋）を抽出中...")
    for path in tqdm(images_name):
        res = extract_noise_residual(path, denoising_function, crop_size, crop_location)
        norm = np.linalg.norm(res)
        if norm > 0:
            res_normalized = res / norm
        else:
            res_normalized = np.zeros_like(res)

        residuals.append(res_normalized)

    print("Step 2/2: 総当たりで類似度（内積マトリクス）を計算中...")
    for i in tqdm(range(num_images)):
        for j in range(i + 1, num_images):
            similarity = np.vdot(residuals[i], residuals[j])
            C[i, j] = similarity
            C[j, i] = similarity

    return C

# =========================================================
# Step 2: 初期クラスタリング (AL-ICM & WEAC)
# =========================================================
def al_icm(w, initial_labels=None, limit=100):
    n = w.shape[0]
    w_clean = w.copy()
    np.fill_diagonal(w_clean, 0)

    if initial_labels is None or len(initial_labels) != n:
        labels = np.ones(n, dtype=int)
    else:
        _, labels = np.unique(initial_labels, return_inverse=True)

    for itr in range(limit):
        prev_labels = labels.copy()
        unique_labels = np.unique(labels)
        max_label = unique_labels.max()

        for i in range(n):
            best_label = labels[i]
            max_affinity = -np.inf

            candidate_labels = list(unique_labels)
            candidate_labels.append(max_label + 1)

            for c in candidate_labels:
                affinity = np.sum(w_clean[i, labels == c])
                if affinity > max_affinity:
                    max_affinity = affinity
                    best_label = c

            labels[i] = best_label
            if best_label > max_label:
                unique_labels = np.unique(labels)
                max_label = unique_labels.max()

        if itr % 15 == 0:
            _, labels = np.unique(labels, return_inverse=True)

        if np.array_equal(prev_labels, labels):
            break

    _, labels = np.unique(labels, return_inverse=True)
    return labels

def weac_python(base_cls, ncai, cls_nums):
    n_samples, n_base = base_cls.shape
    ncai = ncai.flatten()
    weight_sum = np.sum(ncai)

    S = np.zeros((n_samples, n_samples))
    for k in range(n_base):
        labels = base_cls[:, k]
        match = (labels[:, None] == labels[None, :])
        S += match.astype(float) * ncai[k]

    S = S / weight_sum
    dist_matrix = 1 - S
    dist_vector = squareform(dist_matrix, checks=False)

    Z = linkage(dist_vector, method='average')

    results = {}
    for k in cls_nums:
        results[k] = fcluster(Z, t=k, criterion='maxclust')

    return results

# =========================================================
# Step 3: 洗練ステップ (Iterative Refinement)
# =========================================================
def refine_clustering_labels(C, initial_labels, max_iter=20, tol=1e-4):
    n_samples = C.shape[0]
    refined_labels = initial_labels.copy()

    C_clean = C.copy()
    np.fill_diagonal(C_clean, 0)

    for iteration in range(max_iter):
        old_labels = refined_labels.copy()
        unique_labels = np.unique(refined_labels)

        for i in range(n_samples):
            best_cluster = refined_labels[i]
            max_score = -np.inf

            for k in unique_labels:
                cluster_mask = (refined_labels == k)
                cluster_mask[i] = False

                cluster_size = np.sum(cluster_mask)

                if cluster_size > 0:
                    score = np.sum(C_clean[i, cluster_mask]) / cluster_size
                else:
                    score = 0

                if score > max_score:
                    max_score = score
                    best_cluster = k

            refined_labels[i] = best_cluster

        changed_ratio = np.sum(old_labels != refined_labels) / n_samples

        if changed_ratio <= tol:
            break

    return refined_labels
