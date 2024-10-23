import torch
print("Hello")
if __name__ == '__main__':
    print(torch.__version__)
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is not available")
    # import torch
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    import torch_geometric
    print(torch_geometric.__version__)
    import torch
    import torch_cluster

    x = torch.rand((100, 3), device='cuda')
    batch = torch.zeros(100, dtype=torch.long, device='cuda')
    ratio = 0.5

    idx = torch_cluster.fps(x, batch, ratio)
    print("FPS indices:", idx)