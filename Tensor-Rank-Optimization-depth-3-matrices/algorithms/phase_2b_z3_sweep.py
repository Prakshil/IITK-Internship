import numpy as np
import z3
import argparse
import time
import os
import sys

# Add current directory to path to import sa_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from sa_utils import T_target
except ImportError:
    # Fallback if sa_utils is not available directly
    def build_target_f2():
        T = np.zeros((9, 9, 9), dtype=np.int8)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    T[i*3+j, j*3+k, i*3+k] = 1
        return T
    T_target = build_target_f2()

def z3_sweep(checkpoint_path, max_radius=15, timeout_ms=300000):
    print(f"Loading checkpoint from {checkpoint_path}...")
    data = np.load(checkpoint_path)
    U_base, V_base, W_base = data['U'], data['V'], data['W']
    col_drop = data['col_drop']
    base_error = data['error']
    
    print(f"Checkpoint Base Error: {base_error}/729 (col_drop={col_drop})")
    
    rank = U_base.shape[1]
    
    # Create Z3 solver
    solver = z3.Solver()
    
    # Variables
    U_vars = [[z3.Bool(f"U_{i}_{r}") for r in range(rank)] for i in range(9)]
    V_vars = [[z3.Bool(f"V_{j}_{r}") for r in range(rank)] for j in range(9)]
    W_vars = [[z3.Bool(f"W_{k}_{r}") for r in range(rank)] for k in range(9)]
    
    print("Building Z3 constraints (this may take a moment)...")
    # Tensor reconstruction logic in Z3
    for i in range(9):
        for j in range(9):
            for k in range(9):
                target_val = bool(T_target[i, j, k])
                
                # Z3 formula for this cell
                terms = [z3.And(U_vars[i][r], V_vars[j][r], W_vars[k][r]) for r in range(rank)]
                # XOR sum
                cell_sum = terms[0]
                for r in range(1, rank):
                    cell_sum = z3.Xor(cell_sum, terms[r])
                
                # Constraint
                solver.add(cell_sum == target_val)
                
    # Function to count bit flips from base
    def diff_term(vars_matrix, base_matrix):
        diffs = []
        for i in range(vars_matrix.shape[0]):
            for r in range(vars_matrix.shape[1]):
                base_bit = bool(base_matrix[i, r])
                var = vars_matrix[i][r]
                diff = z3.Not(var) if base_bit else var
                diffs.append(z3.If(diff, 1, 0))
        return diffs
        
    diffs_U = diff_term(np.array(U_vars), U_base)
    diffs_V = diff_term(np.array(V_vars), V_base)
    diffs_W = diff_term(np.array(W_vars), W_base)
    
    total_flips = z3.Sum(diffs_U + diffs_V + diffs_W)
    
    print("Variables and base constraints added.")
    
    for radius in range(1, max_radius + 1):
        print(f"\n--- Trying Radius {radius} ---")
        
        # We push a new context for the radius constraint
        solver.push()
        solver.add(total_flips <= radius)
        
        solver.set("timeout", timeout_ms)
        
        start_t = time.time()
        res = solver.check()
        elapsed = time.time() - start_t
        
        if res == z3.sat:
            print(f"SATISFIABLE at radius {radius}! Solved in {elapsed:.1f}s.")
            model = solver.model()
            
            # Extract matrices
            U_new = np.zeros_like(U_base)
            V_new = np.zeros_like(V_base)
            W_new = np.zeros_like(W_base)
            
            for i in range(9):
                for r in range(rank):
                    U_new[i, r] = 1 if z3.is_true(model[U_vars[i][r]]) else 0
                    V_new[i, r] = 1 if z3.is_true(model[V_vars[i][r]]) else 0
                    W_new[i, r] = 1 if z3.is_true(model[W_vars[i][r]]) else 0
            
            out_file = f"success_rank{rank}_f2_drop{col_drop}.npz"
            np.savez(out_file, U=U_new, V=V_new, W=W_new)
            print(f"Saved exact 0-error solution to {out_file}!")
            return True
            
        elif res == z3.unsat:
            print(f"UNSAT at radius {radius} (took {elapsed:.1f}s).")
            solver.pop() # Remove radius constraint to try next radius
        else:
            print(f"UNKNOWN/TIMEOUT at radius {radius} (took {elapsed:.1f}s).")
            solver.pop() # Pop and continue, although usually if it times out, larger will timeout too
            
    print(f"\nSweep completed up to radius {max_radius} without finding a solution.")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z3 SAT Solver Sweep for Phase 2")
    parser.add_argument("--checkpoint", type=str, default="sa_checkpoint.npz", help="Path to checkpoint .npz")
    parser.add_argument("--max-radius", type=int, default=15, help="Maximum Hamming radius to search")
    parser.add_argument("--timeout", type=int, default=300000, help="Timeout per radius in milliseconds")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)
        
    z3_sweep(args.checkpoint, args.max_radius, args.timeout)
