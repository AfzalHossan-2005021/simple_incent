import os
import ot
import time
import torch
import datetime
import numpy as np
import pandas as pd

from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from typing import Optional, Tuple, Union
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture

from .utils import fused_gromov_wasserstein_incent, to_dense_array, extract_data_matrix, jensenshannon_divergence_backend, pairwise_msd


def _safe_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-normalize a matrix while avoiding division-by-zero."""
    X = np.asarray(X, dtype=np.float64)
    den = X.sum(axis=1, keepdims=True)
    den = np.maximum(den, eps)
    return X / den


def _dense_expr_matrix(curr_slice: AnnData, use_rep: Optional[str]) -> np.ndarray:
    """Extract dense feature matrix for clustering/portion profiling."""
    X = to_dense_array(extract_data_matrix(curr_slice, use_rep))
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {X.shape}.")
    return X


def _celltype_one_hot(labels: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """One-hot encoding for cell type labels against a fixed category set."""
    cat_to_id = {c: i for i, c in enumerate(categories)}
    out = np.zeros((labels.shape[0], len(categories)), dtype=np.float64)
    for i, lab in enumerate(labels):
        j = cat_to_id.get(lab, None)
        if j is not None:
            out[i, j] = 1.0
    return out


def _discover_portions(
    curr_slice: AnnData,
    use_rep: Optional[str],
    max_portions: int,
    min_portion_size: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Discover an unknown number of anatomical portions in a slice using spatial + biological features.

    Returns
    -------
    labels : (n_cells,) hard portion labels
    probs : (n_cells, k) soft membership probabilities
    k : int selected number of portions
    """
    n = curr_slice.shape[0]
    if n == 0:
        raise ValueError("Cannot discover portions on an empty slice.")

    if n < max(2 * min_portion_size, 30):
        labels = np.zeros(n, dtype=np.int64)
        probs = np.ones((n, 1), dtype=np.float64)
        return labels, probs, 1

    coords = np.asarray(curr_slice.obsm['spatial'], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("Expected curr_slice.obsm['spatial'] to be an (n,2+) array.")

    expr = _dense_expr_matrix(curr_slice, use_rep)
    expr = np.log1p(np.maximum(expr, 0.0))
    expr = expr - expr.mean(axis=0, keepdims=True)
    expr_std = expr.std(axis=0, keepdims=True)
    expr_std = np.where(expr_std < 1e-8, 1.0, expr_std)
    expr = expr / expr_std

    labels = np.asarray(curr_slice.obs['cell_type_annot'].astype(str).values)
    uniq_types = np.array(sorted(pd.Index(labels).unique()), dtype=object)
    onehot = _celltype_one_hot(labels, uniq_types)

    # Weighted concatenation. Spatial drives topology; bio features disambiguate adjacent regions.
    feat = np.concatenate([
        coords / (coords.std(axis=0, keepdims=True) + 1e-8),
        0.75 * onehot,
        0.20 * expr,
    ], axis=1)

    k_upper = int(max(1, min(max_portions, n // max(1, min_portion_size))))
    if k_upper <= 1:
        labels = np.zeros(n, dtype=np.int64)
        probs = np.ones((n, 1), dtype=np.float64)
        return labels, probs, 1

    best = None
    best_score = np.inf

    for k in range(1, k_upper + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                reg_covar=1e-5,
                random_state=random_state,
                n_init=3,
                max_iter=300,
            )
            gmm.fit(feat)
            pred = gmm.predict(feat)
            counts = np.bincount(pred, minlength=k)

            # Reject pathological tiny components that often represent noise slivers.
            if k > 1 and np.any(counts < min_portion_size):
                continue

            bic = gmm.bic(feat)
            # Light complexity penalty to avoid over-segmentation in noisy slices.
            score = bic + 3.0 * k * np.log(n + 1.0)
            if score < best_score:
                best_score = score
                best = gmm
        except Exception:
            continue

    if best is None:
        labels = np.zeros(n, dtype=np.int64)
        probs = np.ones((n, 1), dtype=np.float64)
        return labels, probs, 1

    probs = best.predict_proba(feat).astype(np.float64)
    labels = probs.argmax(axis=1).astype(np.int64)
    return labels, probs, int(probs.shape[1])


def _portion_celltype_profiles(
    labels: np.ndarray,
    probs: np.ndarray,
    celltypes: np.ndarray,
    global_types: np.ndarray,
) -> np.ndarray:
    """Compute soft cell-type profiles per discovered portion."""
    onehot = _celltype_one_hot(celltypes, global_types)
    profiles = probs.T @ onehot
    profiles = _safe_normalize_rows(profiles)
    return profiles


def _jsd_rows(P: np.ndarray, Q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Pairwise Jensen-Shannon distance between row distributions of P and Q."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = _safe_normalize_rows(P + eps)
    Q = _safe_normalize_rows(Q + eps)

    out = np.zeros((P.shape[0], Q.shape[0]), dtype=np.float64)
    for i in range(P.shape[0]):
        p = P[i][None, :]
        m = 0.5 * (p + Q)
        kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)), axis=1)
        kl_qm = np.sum(Q * (np.log(Q + eps) - np.log(m + eps)), axis=1)
        out[i, :] = np.sqrt(np.maximum(0.5 * (kl_pm + kl_qm), 0.0))
    return out


def _portion_constraint_penalty(
    sliceA: AnnData,
    sliceB: AnnData,
    use_rep: Optional[str],
    max_portions: int,
    min_portion_size: int,
    random_state: int,
    relation_threshold: float,
) -> Tuple[np.ndarray, dict]:
    """
    Build an organ-agnostic, soft portion-compatibility penalty matrix (nA x nB).

    Lower penalty means higher compatibility. Supports asymmetric and partial overlap.
    """
    labsA = np.asarray(sliceA.obs['cell_type_annot'].astype(str).values)
    labsB = np.asarray(sliceB.obs['cell_type_annot'].astype(str).values)
    global_types = np.array(sorted(pd.Index(labsA).union(pd.Index(labsB))), dtype=object)

    _, probsA, kA = _discover_portions(
        curr_slice=sliceA,
        use_rep=use_rep,
        max_portions=max_portions,
        min_portion_size=min_portion_size,
        random_state=random_state,
    )
    _, probsB, kB = _discover_portions(
        curr_slice=sliceB,
        use_rep=use_rep,
        max_portions=max_portions,
        min_portion_size=min_portion_size,
        random_state=random_state + 17,
    )

    if kA == 1 and kB == 1:
        penalty = np.zeros((sliceA.shape[0], sliceB.shape[0]), dtype=np.float64)
        meta = {
            'kA': 1,
            'kB': 1,
            'portion_transport': np.array([[1.0]], dtype=np.float64),
            'portion_cost': np.array([[0.0]], dtype=np.float64),
            'allowed_mask': np.array([[1.0]], dtype=np.float64),
        }
        return penalty, meta

    profA = _portion_celltype_profiles(np.argmax(probsA, axis=1), probsA, labsA, global_types)
    profB = _portion_celltype_profiles(np.argmax(probsB, axis=1), probsB, labsB, global_types)
    portion_cost = _jsd_rows(profA, profB)

    massA = probsA.sum(axis=0).astype(np.float64)
    massB = probsB.sum(axis=0).astype(np.float64)
    massA = massA / np.maximum(massA.sum(), 1e-12)
    massB = massB / np.maximum(massB.sum(), 1e-12)

    # Portion-level OT permits asymmetric overlap via transport plan on discovered portions.
    T = ot.emd(massA, massB, portion_cost)
    T = np.asarray(T, dtype=np.float64)
    if T.sum() > 0:
        T = T / T.sum()

    # Allowed portion-pairs are those that carry enough mass in the portion-level transport.
    tau = relation_threshold * max(float(T.max()), 1e-12)
    allowed = (T >= tau).astype(np.float64)
    if allowed.sum() == 0:
        allowed[np.unravel_index(np.argmax(T), T.shape)] = 1.0

    # Soft pairwise compatibility from soft memberships through allowed portion correspondences.
    allowed_prob = probsA @ allowed @ probsB.T
    penalty = 1.0 - np.clip(allowed_prob, 0.0, 1.0)

    meta = {
        'kA': int(kA),
        'kB': int(kB),
        'portion_transport': T,
        'portion_cost': portion_cost,
        'allowed_mask': allowed,
    }
    return penalty, meta


def pairwise_align(
    sliceA: AnnData, 
    sliceB: AnnData, 
    alpha: float,
    beta: float,
    gamma: float,
    radius: float,
    filePath: str,
    use_rep: Optional[str] = None, 
    G_init = None, 
    a_distribution = None, 
    b_distribution = None, 
    norm: bool = False, 
    numItermax: int = 6000, 
    backend = ot.backend.NumpyBackend(), 
    use_gpu: bool = False, 
    return_obj: bool = False,
    verbose: bool = False, 
    gpu_verbose: bool = True, 
    sliceA_name: Optional[str] = None,
    sliceB_name: Optional[str] = None,
    overwrite = False,
    neighborhood_dissimilarity: str='jsd',
    dummy_cell: bool = True,
    portion_mode: str = 'off',
    portion_penalty_weight: float = 2.0,
    portion_hard_mask: bool = False,
    max_portions: int = 8,
    min_portion_size: int = 40,
    portion_relation_threshold: float = 0.05,
    random_state: int = 0,
    **kwargs) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float, float, float, float]]:
    """

    This method is written by Anup Bhowmik, CSE, BUET

    Calculates and returns optimal alignment of two slices of single cell MERFISH data. 
    
    Args:
        sliceA: Slice A to align.
        sliceB: Slice B to align.
        alpha:  weight for spatial distance
        gamma: weight for gene expression distance (JSD)
        beta: weight for cell type one hot encoding
        radius: radius for cellular neighborhood
        portion_mode: ``'off'`` or ``'auto'``. In auto mode, discovers an unknown number of portions
                  in each slice and constrains transport to compatible portions.
        portion_penalty_weight: Weight of soft portion incompatibility penalty (used when portion_mode='auto').
        portion_hard_mask: If ``True``, imposes near-hard blocking of incompatible portion pairs.
        max_portions: Upper bound for automatically discovered portions per slice.
        min_portion_size: Minimum cell count per discovered portion.
        portion_relation_threshold: Threshold for considering a portion pair compatible relative to
                       max portion-level transport mass.
        random_state: Random seed used in automatic portion discovery.

        dissimilarity: Expression dissimilarity measure: ``'kl'`` or ``'euclidean'``.
        use_rep: If ``None``, uses ``slice.X`` to calculate dissimilarity between spots, otherwise uses the representation given by ``slice.obsm[use_rep]``.
        G_init (array-like, optional): Initial mapping to be used in FGW-OT, otherwise default is uniform mapping.
        a_distribution (array-like, optional): Distribution of sliceA spots, otherwise default is uniform.
        b_distribution (array-like, optional): Distribution of sliceB spots, otherwise default is uniform.
        numItermax: Max number of iterations during FGW-OT.
        norm: If ``True``, scales spatial distances such that neighboring spots are at distance 1. Otherwise, spatial distances remain unchanged.
        backend: Type of backend to run calculations. For list of backends available on system: ``ot.backend.get_backend_list()``.
        use_gpu: If ``True``, use gpu. Otherwise, use cpu. Currently we only have gpu support for Pytorch.
        return_obj: If ``True``, additionally returns objective function output of FGW-OT.
        verbose: If ``True``, FGW-OT is verbose.
        gpu_verbose: If ``True``, print whether gpu is being used to user.
   
    Returns:
        - Alignment of spots.

        If ``return_obj = True``, additionally returns:
        
        - Objective function output of cost 
    """

    start_time = time.time()

    if not os.path.exists(filePath):
        os.makedirs(filePath)

    logFile = open(f"{filePath}/log.txt", "w")

    logFile.write(f"pairwise_align_INCENT\n")
    currDateTime = datetime.datetime.now()

    # logFile.write(f"{currDateTime.date()}, {currDateTime.strftime("%I:%M %p")} BDT, {currDateTime.strftime("%A")} \n")

    logFile.write(f"{currDateTime}\n")
    logFile.write(f"sliceA_name: {sliceA_name}, sliceB_name: {sliceB_name}\n")
   

    logFile.write(f"alpha: {alpha}\n")
    logFile.write(f"beta: {beta}\n")
    logFile.write(f"gamma: {gamma}\n")
    logFile.write(f"radius: {radius}\n")
    logFile.write(f"portion_mode: {portion_mode}\n")
    logFile.write(f"portion_penalty_weight: {portion_penalty_weight}\n")
    logFile.write(f"portion_hard_mask: {portion_hard_mask}\n")


    
    # Determine if gpu or cpu is being used
    if use_gpu:
        if isinstance(backend,ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            print("We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu.")
            use_gpu = False
    else:
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")

    if not torch.cuda.is_available():
        use_gpu = False
        print("CUDA is not available on your system. Reverting to CPU.")
    
    # check if slices are valid
    for s in [sliceA, sliceB]:
        if not len(s):
            raise ValueError(f"Found empty `AnnData`:\n{s}.")

    
    # Backend
    nx = backend

    # Filter to shared genes
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two slices.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]


    # Filter to shared cell types
    # This is needed for the cell-type mismatch penalty, and also ensures that the neighborhood distributions are comparable (same set of cell types).
    shared_cell_types = pd.Index(sliceA.obs['cell_type_annot']).unique().intersection(pd.Index(sliceB.obs['cell_type_annot']).unique())
    if len(shared_cell_types) == 0:
        raise ValueError("No shared cell types between the two slices.")
    sliceA = sliceA[sliceA.obs['cell_type_annot'].isin(shared_cell_types)]
    sliceB = sliceB[sliceB.obs['cell_type_annot'].isin(shared_cell_types)]

    
    # Calculate spatial distances
    coordinatesA = sliceA.obsm['spatial'].copy()
    coordinatesB = sliceB.obsm['spatial'].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = nx.from_numpy(coordinatesB)
    
    if isinstance(nx,ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA,coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB,coordinatesB, metric='euclidean')

    # print the shape of D_A and D_B
    # print("D_A.shape: ", D_A.shape)
    # print("D_B.shape: ", D_B.shape)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()


    # Calculate gene expression dissimilarity
    # filePath = '/content/drive/MyDrive/Thesis_data_anup/local_data'
    cosine_dist_gene_expr = cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = use_rep, use_gpu = use_gpu, nx = nx, beta = beta, overwrite=overwrite)

    # ── Explicit cell-type mismatch penalty ──────────────────────────────
    # Binary matrix: 0 for same type, 1 for different type.
    # Added to M1 so it enters the FW gradient directly → strong cell-type signal.

    _lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    _lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    M_celltype = (_lab_A[:, None] != _lab_B[None, :]).astype(np.float64)
    M1_combined = (1 - beta) * cosine_dist_gene_expr + beta * M_celltype
    logFile.write(f"[cell_type_penalty] beta={beta}, M_celltype shape={M_celltype.shape}\n")

    # ── Organ-agnostic portion-aware constraint ─────────────────────────
    # Auto-discovers unknown number of portions and discourages transport
    # across incompatible portions. This supports asymmetric and partial overlap.
    if portion_mode not in {'off', 'auto'}:
        raise ValueError(f"Invalid portion_mode {portion_mode!r}. Expected one of {'off','auto'}.")

    if portion_mode == 'auto':
        portion_penalty, portion_meta = _portion_constraint_penalty(
            sliceA=sliceA,
            sliceB=sliceB,
            use_rep=use_rep,
            max_portions=max_portions,
            min_portion_size=min_portion_size,
            random_state=random_state,
            relation_threshold=portion_relation_threshold,
        )

        M1_combined = M1_combined + portion_penalty_weight * portion_penalty

        if portion_hard_mask:
            # Apply strong additive barrier where compatibility is near zero.
            hard_mask = portion_penalty > 0.999
            barrier = 10.0 * max(float(np.max(M1_combined)), 1.0)
            M1_combined = M1_combined + hard_mask.astype(np.float64) * barrier

        logFile.write(
            f"[portion_constraint] kA={portion_meta['kA']}, kB={portion_meta['kB']}, "
            f"penalty_mean={float(np.mean(portion_penalty)):.6f}, "
            f"penalty_max={float(np.max(portion_penalty)):.6f}\n"
        )
        print(
            f"[portion_constraint] auto mode enabled: kA={portion_meta['kA']}, "
            f"kB={portion_meta['kB']}, penalty_mean={float(np.mean(portion_penalty)):.6f}"
        )


    M1 = nx.from_numpy(M1_combined)


    # jensenshannon_divergence_backend actually returns jensen shannon distance
    # neighborhood_distribution_slice_1, neighborhood_distribution_slice_1 will be pre computed

    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = np.load(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice A")
        neighborhood_distribution_sliceA = neighborhood_distribution(sliceA, radius = radius)


        neighborhood_distribution_sliceA += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice A")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceA_name}.npy", neighborhood_distribution_sliceA)


    if os.path.exists(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy") and not overwrite:
        print("Loading precomputed neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = np.load(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy")
    else:
        print("Calculating neighborhood distribution of slice B")
        neighborhood_distribution_sliceB = neighborhood_distribution(sliceB, radius = radius)


        neighborhood_distribution_sliceB += 0.01 # for avoiding zero division error
        # print("Saving neighborhood distribution of slice B")
        # np.save(f"{filePath}/neighborhood_distribution_{sliceB_name}.npy", neighborhood_distribution_sliceB)


    if ('numpy' in str(type(neighborhood_distribution_sliceA))) and use_gpu:
        neighborhood_distribution_sliceA = torch.from_numpy(neighborhood_distribution_sliceA)
    if ('numpy' in str(type(neighborhood_distribution_sliceB))) and use_gpu:
        neighborhood_distribution_sliceB = torch.from_numpy(neighborhood_distribution_sliceB)

    if use_gpu:
        neighborhood_distribution_sliceA = neighborhood_distribution_sliceA.cuda()
        neighborhood_distribution_sliceB = neighborhood_distribution_sliceB.cuda()

    if neighborhood_dissimilarity == 'jsd':
        if os.path.exists(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy") and not overwrite:
            print("Loading precomputed JSD of neighborhood distribution for slice A and slice B")
            js_dist_neighborhood = np.load(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy")
            
        else:
            print("Calculating JSD of neighborhood distribution for slice A and slice B")

            js_dist_neighborhood = jensenshannon_divergence_backend(neighborhood_distribution_sliceA, neighborhood_distribution_sliceB)

            if isinstance(js_dist_neighborhood, torch.Tensor):
                js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()

            # print("Saving precomputed JSD of neighborhood distribution for slice A and slice B")
            # np.save(f"{filePath}/js_dist_neighborhood_{sliceA_name}_{sliceB_name}.npy", js_dist_neighborhood)
  
        M2 = nx.from_numpy(js_dist_neighborhood)

    elif neighborhood_dissimilarity == 'cosine':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor) or isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndA = neighborhood_distribution_sliceA
            ndB = neighborhood_distribution_sliceB
            if not isinstance(ndA, torch.Tensor):
                ndA = torch.from_numpy(np.asarray(ndA))
            if not isinstance(ndB, torch.Tensor):
                ndB = torch.from_numpy(np.asarray(ndB))
            if use_gpu:
                ndA = ndA.cuda()
                ndB = ndB.cuda()
            numerator = ndA @ ndB.T
            denom = ndA.norm(dim=1)[:, None] * ndB.norm(dim=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
            cosine_dist_neighborhood = cosine_dist_neighborhood.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
            ndB = np.asarray(neighborhood_distribution_sliceB)
            numerator = ndA @ ndB.T
            denom = np.linalg.norm(ndA, axis=1)[:, None] * np.linalg.norm(ndB, axis=1)[None, :]
            cosine_dist_neighborhood = 1 - numerator / denom
        M2 = nx.from_numpy(cosine_dist_neighborhood)

    elif neighborhood_dissimilarity == 'msd':
        if isinstance(neighborhood_distribution_sliceA, torch.Tensor):
            ndA = neighborhood_distribution_sliceA.detach().cpu().numpy()
        else:
            ndA = np.asarray(neighborhood_distribution_sliceA)
        if isinstance(neighborhood_distribution_sliceB, torch.Tensor):
            ndB = neighborhood_distribution_sliceB.detach().cpu().numpy()
        else:
            ndB = np.asarray(neighborhood_distribution_sliceB)

        msd_neighborhood = pairwise_msd(ndA, ndB)
        M2 = nx.from_numpy(msd_neighborhood)

    else:
        raise ValueError(
            "Invalid neighborhood_dissimilarity. Expected one of {'jsd','cosine','msd'}; "
            f"got {neighborhood_dissimilarity!r}."
        )

    # ── Dummy cell augmentation ────────────────────────────────────────────
    # Only add a dummy on the side that actually has a deficit.
    # _has_dummy_src: True if source has fewer cells → need dummy source (birth)
    # _has_dummy_tgt: True if target has fewer cells → need dummy target (death)
    _has_dummy_src = False
    _has_dummy_tgt = False

    if dummy_cell:
        from collections import Counter
        ns, nt = sliceA.shape[0], sliceB.shape[0]
        labels_A = sliceA.obs['cell_type_annot'].values
        labels_B = sliceB.obs['cell_type_annot'].values
        counts_A = Counter(labels_A)
        counts_B = Counter(labels_B)
        all_types = set(counts_A.keys()) | set(counts_B.keys())
        _budget = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)
        _w_dummy_src = _budget - ns   # extra weight for dummy source cell (birth)
        _w_dummy_tgt = _budget - nt   # extra weight for dummy target cell (death)
        _has_dummy_src = _w_dummy_src > 0
        _has_dummy_tgt = _w_dummy_tgt > 0

        logFile.write(f"[dummy_cell] budget={_budget}, w_dummy_src={_w_dummy_src}, w_dummy_tgt={_w_dummy_tgt}\n")
        print(f"[dummy_cell] budget={_budget}, "
              f"dummy_src={'YES (birth)' if _has_dummy_src else 'NO'} w={_w_dummy_src}, "
              f"dummy_tgt={'YES (death)' if _has_dummy_tgt else 'NO'} w={_w_dummy_tgt}")

        def _to_np(x):
            """Convert backend tensor to float64 numpy."""
            try:
                return x.cpu().detach().numpy().astype(np.float64)
            except Exception:
                return np.array(x, dtype=np.float64)

        _ns_aug = ns + (1 if _has_dummy_src else 0)
        _nt_aug = nt + (1 if _has_dummy_tgt else 0)

        # ---- Augment D_A if dummy source needed ----
        # Dummy spatial distance = 0: makes dummy INVISIBLE to the Gromov term.
        # Since C1[dummy, k] = 0 for all k, the gradient df(G) for real cells
        # is identical to the non-dummy case → spatial alignment preserved.
        if _has_dummy_src:
            D_A_np = _to_np(D_A)
            D_A_aug = np.zeros((_ns_aug, _ns_aug), dtype=np.float64)
            D_A_aug[:ns, :ns] = D_A_np
            # D_A_aug[ns, :] and D_A_aug[:, ns] stay 0
            D_A = nx.from_numpy(D_A_aug)
            if isinstance(nx, ot.backend.TorchBackend):
                D_A = D_A.float()

        # ---- Augment D_B if dummy target needed ----
        if _has_dummy_tgt:
            D_B_np = _to_np(D_B)
            D_B_aug = np.zeros((_nt_aug, _nt_aug), dtype=np.float64)
            D_B_aug[:nt, :nt] = D_B_np
            # D_B_aug[nt, :] and D_B_aug[:, nt] stay 0
            D_B = nx.from_numpy(D_B_aug)
            if isinstance(nx, ot.backend.TorchBackend):
                D_B = D_B.float()

        # ---- Precompute per-type max cost from the same-type submatrix ----
        M1_np = _to_np(M1)
        M2_np = _to_np(M2)
        _type_M1_max = {}
        _type_M2_max = {}
        for _type_k in all_types:
            _S = np.where(_lab_A == _type_k)[0]
            _T = np.where(_lab_B == _type_k)[0]
            if len(_S) > 0 and len(_T) > 0:
                _type_M1_max[_type_k] = float(M1_np[np.ix_(_S, _T)].max())
                _type_M2_max[_type_k] = float(M2_np[np.ix_(_S, _T)].max())
            else:
                # Type absent on one side — fall back to global max
                _type_M1_max[_type_k] = float(M1_np.max())
                _type_M2_max[_type_k] = float(M2_np.max())

        # Per-cell dummy costs: each cell pays the max of its own type's submatrix
        _death_M1 = np.array([_type_M1_max[_lab_A[i]] for i in range(ns)], dtype=np.float64)
        _birth_M1 = np.array([_type_M1_max[_lab_B[j]] for j in range(nt)], dtype=np.float64)
        _death_M2 = np.array([_type_M2_max[_lab_A[i]] for i in range(ns)], dtype=np.float64)
        _birth_M2 = np.array([_type_M2_max[_lab_B[j]] for j in range(nt)], dtype=np.float64)

        epsilon = 1e-6

        # ---- Augment M1: add dummy row/col only where needed ----
        M1_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
        M1_aug[:ns, :nt] = M1_np
        if _has_dummy_tgt:
            M1_aug[:ns, nt] = _death_M1 + epsilon   # src cell i → dummy tgt: max within i's type
        if _has_dummy_src:
            M1_aug[ns, :nt] = _birth_M1 + epsilon   # dummy src → tgt cell j: max within j's type
        if _has_dummy_src and _has_dummy_tgt:
            M1_aug[ns, nt] = np.max(M1_np) + epsilon          # dummy-to-dummy is free
        M1 = nx.from_numpy(M1_aug)

        # ---- Augment M2: add dummy row/col only where needed ----
        M2_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
        M2_aug[:ns, :nt] = M2_np
        if _has_dummy_tgt:
            M2_aug[:ns, nt] = _death_M2 + epsilon   # src cell i → dummy tgt: max within i's type
        if _has_dummy_src:
            M2_aug[ns, :nt] = _birth_M2 + epsilon   # dummy src → tgt cell j: max within j's type
        if _has_dummy_src and _has_dummy_tgt:
            M2_aug[ns, nt] = np.max(M2_np) + epsilon          # dummy-to-dummy is free
        M2 = nx.from_numpy(M2_aug)

        logFile.write(f"[dummy_cell] Augmented: D_A {tuple(D_A.shape)}, D_B {tuple(D_B.shape)}, "
                      f"M1 {tuple(M1.shape)}, M2 {tuple(M2.shape)}\n")

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        # M = M.cuda()

        M1 = M1.cuda()
        M2 = M2.cuda()
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            D_A = D_A.cuda()
            D_B = D_B.cuda()
    
    # init distributions
    if a_distribution is None:
        if dummy_cell:
            if _has_dummy_src:
                a_vals = np.full(ns + 1, 1.0 / _budget, dtype=np.float64)
                a_vals[-1] = float(_w_dummy_src) / _budget
                a = nx.from_numpy(a_vals)
            else:
                a = nx.ones((ns,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            # uniform distribution, a = array([1/n, 1/n, ...])
            a = nx.ones((sliceA.shape[0],))/sliceA.shape[0]
    else:
        if dummy_cell:
            raise ValueError("Custom a_distribution is not supported with dummy_cell=True.")
        a = nx.from_numpy(a_distribution)
        
    if b_distribution is None:
        if dummy_cell:
            if _has_dummy_tgt:
                b_vals = np.full(nt + 1, 1.0 / _budget, dtype=np.float64)
                b_vals[-1] = float(_w_dummy_tgt) / _budget
                b = nx.from_numpy(b_vals)
            else:
                b = nx.ones((nt,), type_as=np.array([1.0], dtype=np.float64)) / _budget
        else:
            b = nx.ones((sliceB.shape[0],))/sliceB.shape[0]
    else:
        if dummy_cell:
            raise ValueError("Custom b_distribution is not supported with dummy_cell=True.")
        b = nx.from_numpy(b_distribution)

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()
    
    if norm:
        D_A /= nx.min(D_A[D_A>0])
        D_B /= nx.min(D_B[D_B>0])
    
    # Run OT
    if G_init is not None:
        if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
            # Pad user-provided (ns x nt) G_init to augmented dims
            _gi = np.array(G_init, dtype=np.float64)
            _gi_aug = np.zeros((_ns_aug, _nt_aug), dtype=np.float64)
            _gi_aug[:ns, :nt] = _gi
            G_init = _gi_aug
        G_init = nx.from_numpy(G_init)
        if isinstance(nx,ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init = G_init.cuda()

    if dummy_cell:
        # Use original dims for initial objective logging
        _ns_log, _nt_log = sliceA.shape[0], sliceB.shape[0]
        G = np.ones((_ns_log, _nt_log)) / (_ns_log * _nt_log)
    else:
        G = np.ones((a.shape[0], b.shape[0])) / (a.shape[0] * b.shape[0])

    if neighborhood_dissimilarity == 'jsd':
        initial_obj_neighbor = np.sum(js_dist_neighborhood*G)
    if neighborhood_dissimilarity == 'msd':
        initial_obj_neighbor = np.sum(msd_neighborhood*G)
    elif neighborhood_dissimilarity == 'cosine':
        initial_obj_neighbor = np.sum(cosine_dist_neighborhood*G)

    initial_obj_gene = np.sum(cosine_dist_gene_expr*G)

    if neighborhood_dissimilarity == 'jsd':
        # print(f"Initial objective neighbor (jsd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (jsd): {initial_obj_neighbor}\n")

    elif neighborhood_dissimilarity == 'cosine':
        # print(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor_cos}")
        logFile.write(f"Initial objective neighbor (cosine_dist): {initial_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'msd':
        # print(f"Initial objective neighbor (msd): {initial_obj_neighbor}")
        logFile.write(f"Initial objective neighbor (mean sq distance): {initial_obj_neighbor}\n")

    # print(f"Initial objective gene expr (cosine_dist): {initial_obj_gene}")
    logFile.write(f"Initial objective (cosine_dist): {initial_obj_gene}\n")
    

    # D_A: pairwise dist matrix of sliceA spots coords
    # a: initial distribution(uniform) of sliceA spots
    _fgw_extra = {'numItermaxEmd': 500_000} if dummy_cell else {}
    pi, logw = fused_gromov_wasserstein_incent(M1, M2, D_A, D_B, a, b, G_init = G_init, loss_fun='square_loss', alpha= alpha, gamma=gamma, log=True, numItermax=numItermax,verbose=verbose, use_gpu = use_gpu, **_fgw_extra)
    pi = nx.to_numpy(pi)
    # obj = nx.to_numpy(logw['fgw_dist'])

    # ── Dummy cell: strip dummy row/col, renormalize, report birth/death ────
    if dummy_cell and (_has_dummy_src or _has_dummy_tgt):
        pi_full = pi.copy()

        # Compute birth / death mass before stripping
        birth_mass = float(pi_full[-1, :nt].sum()) if _has_dummy_src else 0.0
        death_mass = float(pi_full[:ns, -1].sum()) if _has_dummy_tgt else 0.0

        # Strip only the dummy row/col that was actually added
        if _has_dummy_src and _has_dummy_tgt:
            pi = pi_full[:ns, :nt]
        elif _has_dummy_src:
            pi = pi_full[:ns, :]      # strip dummy source row only
        elif _has_dummy_tgt:
            pi = pi_full[:, :nt]      # strip dummy target col only

        # Renormalize so pi sums to 1 (fair comparison with balanced baseline)
        pi_sum = pi.sum()
        if pi_sum > 0:
            pi = pi / pi_sum

        logFile.write(f"[dummy_cell] death_mass={death_mass:.6f}, birth_mass={birth_mass:.6f}, "
                      f"real_mass_before_norm={pi_sum:.6f}\n")
        print(f"[dummy_cell] death_mass: {death_mass:.6f}, birth_mass: {birth_mass:.6f}, "
              f"real_to_real_mass: {pi_sum:.6f} (renormalized to 1.0)")

    if neighborhood_dissimilarity == 'jsd':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        jsd_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            jsd_error[i] = pi[i][max_indices[i]] * js_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(jsd_error)
    elif neighborhood_dissimilarity == 'msd':
        final_obj_neighbor = np.sum(msd_neighborhood*pi)

    elif neighborhood_dissimilarity == 'cosine':
        max_indices = np.argmax(pi, axis=1)
        # multiply each value of max_indices from pi_mat with the corresponding js_dist entry
        cos_error = np.zeros(max_indices.shape)
        for i in range(len(max_indices)):
            cos_error[i] = pi[i][max_indices[i]] * cosine_dist_neighborhood[i][max_indices[i]]

        final_obj_neighbor = np.sum(cos_error)


    final_obj_gene = np.sum(cosine_dist_gene_expr * pi)

    if neighborhood_dissimilarity == 'jsd':
        logFile.write(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (jsd): {final_obj_neighbor}\n")
    elif neighborhood_dissimilarity == 'cosine':
        logFile.write(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")
        # print(f"Final objective neighbor (cosine_dist): {final_obj_neighbor}\n")

    logFile.write(f"Final objective gene expr(cosine_dist): {final_obj_gene}\n")
    # print(f"Final objective (cosine_dist): {final_obj_gene}\n")
    

    logFile.write(f"Runtime: {str(time.time() - start_time)} seconds\n")
    # print(f"Runtime: {str(time.time() - start_time)} seconds\n")
    logFile.write(f"---------------------------------------------\n\n\n")

    logFile.close()

    # new code ends

    if isinstance(backend,ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, initial_obj_neighbor, initial_obj_gene, final_obj_neighbor, final_obj_gene
    
    return pi


def neighborhood_distribution(curr_slice, radius):
    """
    This method is added by Anup Bhowmik
    Args:
        curr_slice: Slice to get niche distribution for.
        pairwise_distances: Pairwise distances between cells of a slice.
        radius: Radius of the niche.

    Returns:
        niche_distribution: Niche distribution for the slice.
    """

    # print ("radius", radius)

    unique_cell_types = np.array(list(curr_slice.obs['cell_type_annot'].unique()))
    cell_type_to_index = dict(zip(unique_cell_types, list(range(len(unique_cell_types)))))
    cells_within_radius = np.zeros((curr_slice.shape[0], len(unique_cell_types)), dtype=float)

    # print("time taken for cell type", time_cell_type_end-time_cell_type_start)

    source_coords = curr_slice.obsm['spatial']
    distances = euclidean_distances(source_coords, source_coords)

    for i in tqdm(range(curr_slice.shape[0])):
        # find the indices of the cells within the radius

        target_indices = np.where(distances[i] <= radius)[0]
        # print("i", i)
        # print(target_indices)

        for ind in target_indices:
            cell_type_str_j = str(curr_slice.obs['cell_type_annot'][ind])
            cells_within_radius[i][cell_type_to_index[cell_type_str_j]] += 1

    return np.array(cells_within_radius)


def cosine_distance(sliceA, sliceB, sliceA_name, sliceB_name, filePath, use_rep = None, use_gpu = False, nx = ot.backend.NumpyBackend(), beta = 0.8, overwrite = False):
    from sklearn.metrics.pairwise import cosine_distances
    import os

    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA,use_rep))), nx.from_numpy(to_dense_array(extract_data_matrix(sliceB,use_rep)))

    if isinstance(nx,ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

   
    s_A = A_X + 0.01
    s_B = B_X + 0.01

    fileName = f"{filePath}/cosine_dist_gene_expr_{sliceA_name}_{sliceB_name}.npy"
    
    if os.path.exists(fileName) and not overwrite:
        print("Loading precomputed Cosine distance of gene expression for slice A and slice B")
        cosine_dist_gene_expr = np.load(fileName)
    else:
        print("Calculating cosine dist of gene expression for slice A and slice B")

        # calculate cosine distance manually
        # cosine_dist_gene_expr = 1 - (s_A @ s_B.T) / s_A.norm(dim=1)[:, None] / s_B.norm(dim=1)[None, :]
        # cosine_dist_gene_expr = cosine_dist_gene_expr.cpu().detach().numpy()

        # use sklearn's cosine_distances
        if torch.cuda.is_available():
            s_A = s_A.cpu().detach().numpy()
            s_B = s_B.cpu().detach().numpy()
        cosine_dist_gene_expr = cosine_distances(s_A, s_B)

        print("Saving cosine dist of gene expression for slice A and slice B")
        np.save(fileName, cosine_dist_gene_expr)

    return cosine_dist_gene_expr

