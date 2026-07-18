import multiprocessing
import os
import time
import argparse
import numpy as np
import sys

# Add current directory to path to import sa_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sa_utils import run_chain

def worker(col_drop, iterations, out_dir):
    print(f"Worker started for col_drop={col_drop}")
    run_chain(col_drop, iterations, out_dir)
    # Re-evaluate error from saved file to return
    ckpt_path = os.path.join(out_dir, f"sa_checkpoint_drop{col_drop:02d}.npz")
    if os.path.exists(ckpt_path):
        data = np.load(ckpt_path)
        return col_drop, int(data['error'])
    return col_drop, 729

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1b Multi-chain SA")
    parser.add_argument("--iterations", type=int, default=25_000_000, help="Iterations per chain")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of parallel cores to use")
    parser.add_argument("--out-dir", type=str, default="sa_runs", help="Output directory for checkpoints")
    parser.add_argument("--test", action="store_true", help="Run in test mode (1000 iterations)")
    args = parser.parse_args()
    
    iters = 1000 if args.test else args.iterations
    print(f"Starting Multi-chain SA Orchestrator on {args.cores} cores.")
    print(f"Running {iters} iterations for each of the 23 col_drop values.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    start_t = time.time()
    
    # We want to run col_drop from 0 to 22
    tasks = [(c, iters, args.out_dir) for c in range(23)]
    
    results = []
    with multiprocessing.Pool(args.cores) as pool:
        for res in pool.starmap(worker, tasks):
            results.append(res)
            
    results.sort(key=lambda x: x[1])
    
    print("\n--- MULTI-CHAIN SA RESULTS ---")
    for i, (cd, err) in enumerate(results):
        print(f"Rank {i+1}: col_drop={cd} -> {err}/729 errors")
        
    print(f"Total time elapsed: {time.time() - start_t:.1f}s")
