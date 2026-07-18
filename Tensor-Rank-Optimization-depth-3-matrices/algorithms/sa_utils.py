import numpy as np
import time, math, random, os

def build_target_f2():
    T = np.zeros((9, 9, 9), dtype=np.int8)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                T[i*3+j, j*3+k, i*3+k] = 1
    return T

T_target = build_target_f2()

def calc_f2_tensor(U, V, W):
    recon = np.einsum("ir,jr,kr->ijk", U.astype(np.int32), V.astype(np.int32), W.astype(np.int32))
    return (recon % 2).astype(np.int8)

def calc_error(U, V, W):
    recon = calc_f2_tensor(U, V, W)
    return int((recon != T_target).sum())

def simulated_annealing_f2_with_restart(U_full, V_full, W_full, col_drop,
                                         total_iterations=25_000_000, focus_prob=0.7,
                                         checkpoint_path=None, log_every=1_000_000,
                                         stagnation_threshold=2_000_000):
    start_t = time.time()
    tag = f"[drop={col_drop}]"

    keep = [i for i in range(23) if i != col_drop]
    U = U_full[:, keep].copy()
    V = V_full[:, keep].copy()
    W = W_full[:, keep].copy()
    rank = U.shape[1]

    recon = calc_f2_tensor(U, V, W)
    error_mask = (recon != T_target)
    current_error = int(error_mask.sum())
    violated = set(map(tuple, np.argwhere(error_mask)))

    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_error_global = current_error

    T_start, T_end = 2.0, 0.005
    temp = T_start

    print(f"{tag} Starting | Initial Errors: {current_error}/729")

    i = 0
    restart_count = 0
    steps_since_improvement = 0

    # Cache for weights and cells list to avoid rebuilding every step
    weights_dirty = True
    cells_list = []
    weights = None

    while i < total_iterations:
        remaining = total_iterations - i
        cooling_factor = (T_end / T_start) ** (1.0 / remaining)

        # ---- Adaptive error-biased sampling ----
        if violated and random.random() < focus_prob:
            if weights_dirty:
                cells_list = list(violated)
                u_sums = U.sum(axis=1)
                v_sums = V.sum(axis=1)
                w_sums = W.sum(axis=1)
                
                weights_list = []
                for vi, vj, vk in cells_list:
                    u_touch = int(v_sums[vj]) * int(w_sums[vk])
                    v_touch = int(u_sums[vi]) * int(w_sums[vk])
                    w_touch = int(u_sums[vi]) * int(v_sums[vj])
                    weights_list.append(max(1, u_touch + v_touch + w_touch))
                
                weights = np.array(weights_list, dtype=np.float64)
                weights /= weights.sum()
                weights_dirty = False
                
            vi, vj, vk = cells_list[np.random.choice(len(cells_list), p=weights)]

            r = random.randint(0, rank - 1)
            mat_choice = random.randint(0, 2)
            row = vi if mat_choice == 0 else (vj if mat_choice == 1 else vk)
            col = r
        else:
            mat_choice = random.randint(0, 2)
            row = random.randint(0, 8)
            col = random.randint(0, rank - 1)

        # ---- Compute touch set (delta = which tensor cells change) ----
        if mat_choice == 0:  # flip U[row, col]
            js = np.nonzero(V[:, col])[0]
            ks = np.nonzero(W[:, col])[0]
            idx_i = np.full(js.size * ks.size, row)
            idx_j = np.repeat(js, ks.size)
            idx_k = np.tile(ks, js.size)
        elif mat_choice == 1:  # flip V[row, col]
            is_ = np.nonzero(U[:, col])[0]
            ks = np.nonzero(W[:, col])[0]
            idx_i = np.repeat(is_, ks.size)
            idx_j = np.full(is_.size * ks.size, row)
            idx_k = np.tile(ks, is_.size)
        else:  # flip W[row, col]
            is_ = np.nonzero(U[:, col])[0]
            js = np.nonzero(V[:, col])[0]
            idx_i = np.repeat(is_, js.size)
            idx_j = np.tile(js, is_.size)
            idx_k = np.full(is_.size * js.size, row)

        touched = (idx_i, idx_j, idx_k)

        # ---- Incremental error delta ----
        old_mismatch = error_mask[touched]
        new_recon_vals = 1 - recon[touched]
        new_mismatch = (new_recon_vals != T_target[touched])
        delta_E = int(new_mismatch.sum()) - int(old_mismatch.sum())

        # ---- Metropolis acceptance ----
        accept = delta_E <= 0 or random.random() < math.exp(-delta_E / temp)

        if accept:
            # Apply flip
            if mat_choice == 0:
                U[row, col] ^= 1
            elif mat_choice == 1:
                V[row, col] ^= 1
            else:
                W[row, col] ^= 1

            # Update incremental state
            recon[touched] = new_recon_vals
            for (ii, jj, kk), was_wrong, is_wrong in zip(zip(*touched), old_mismatch, new_mismatch):
                cell = (int(ii), int(jj), int(kk))
                if was_wrong and not is_wrong:
                    violated.discard(cell)
                elif not was_wrong and is_wrong:
                    violated.add(cell)
            error_mask[touched] = new_mismatch
            current_error += delta_E
            weights_dirty = True  # Mark weights cache as dirty

            if current_error < best_error:
                best_error = current_error
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                print(f"{tag} Step {i:9d} | Temp: {temp:.4f} | NEW BEST: {best_error}/729 | Restarts: {restart_count}")
                if checkpoint_path:
                    try:
                        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
                        np.savez(checkpoint_path, U=best_U, V=best_V, W=best_W,
                                 col_drop=col_drop, error=best_error)
                        print(f"Saved checkpoint to {checkpoint_path}")
                    except Exception as e:
                        print(f"Failed saving checkpoint {checkpoint_path}: {e}")
                steps_since_improvement = 0
                if best_error < best_error_global:
                    best_error_global = best_error
            else:
                steps_since_improvement += 1

            if best_error == 0:
                print(f"\n{tag} *** EXACT SOLUTION FOUND AT STEP {i}! ***")
                break

        # ---- Detect stagnation and restart ----
        if steps_since_improvement > stagnation_threshold and i < total_iterations * 0.8:
            restart_count += 1
            print(f"{tag} [RESTART #{restart_count}] Stagnated for {stagnation_threshold} steps, reheating at step {i}...")
            temp = T_start
            steps_since_improvement = 0
            weights_dirty = True
        else:
            temp *= cooling_factor

        if i > 0 and i % log_every == 0:
            print(f"{tag} [Heartbeat] Step {i:9d} | Temp: {temp:.4f} | "
                  f"Current: {current_error}/729 | Best: {best_error}/729 | Violations: {len(violated)}")

        i += 1

    print(f"{tag} Done in {time.time()-start_t:.1f}s | Final Best Error: {best_error}/729 | Restarts: {restart_count}")
    return best_error

def run_chain(col_drop, iterations, out_dir):
    # Reseed so parallel processes don't share RNG state
    random.seed((os.getpid() ^ int(time.time() * 1000) ^ col_drop) & 0xFFFFFFFF)
    np.random.seed((os.getpid() ^ int(time.time() * 1000) ^ col_drop) & 0xFFFFFFFF)

    d = np.load('factorizations_f2.npz', allow_pickle=True)
    scheme = d['3,3,3']
    if isinstance(scheme, np.ndarray) and scheme.ndim == 0:
        scheme = scheme.item()

    if isinstance(scheme, dict):
        U_f2 = np.array(scheme['u'], dtype=np.int8)
        V_f2 = np.array(scheme['v'], dtype=np.int8)
        W_f2 = np.array(scheme['w'], dtype=np.int8)
    else:
        U_f2 = np.array(scheme[0], dtype=np.int8)
        V_f2 = np.array(scheme[1], dtype=np.int8)
        W_f2 = np.array(scheme[2], dtype=np.int8)
    checkpoint_path = os.path.join(out_dir, f"sa_checkpoint_drop{col_drop:02d}.npz")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    print(f"[run_chain] PID={os.getpid()} -> checkpoint_path={checkpoint_path}")
    simulated_annealing_f2_with_restart(U_f2, V_f2, W_f2, col_drop=col_drop,
                                         total_iterations=iterations,
                                         checkpoint_path=checkpoint_path,
                                         stagnation_threshold=2_000_000)
