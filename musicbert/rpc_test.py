import os
import torch
import torch.distributed.rpc as rpc

def test_function(x):
    return x + 1

def main():
    # Set environment variables
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    if args.rank == 0:
        rpc.init_rpc("worker0", rank=0, world_size=2)
        print("Worker 0 initialized")
        rpc.shutdown()
        print("Worker 0 shutdown")
    elif args.rank == 1:
        rpc.init_rpc("worker1", rank=1, world_size=2)
        print("Worker 1 initialized")
        result = rpc.rpc_sync("worker0", test_function, args=(torch.tensor(1),))
        print("RPC result:", result)
        rpc.shutdown()
        print("Worker 1 shutdown")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    main()
