from chimera.utils.reproducibility import set_seed
import torch

def main():
    set_seed()
    print("Hello")
    print(f" Cuda : {torch.cuda.is_available()}")

if __name__ == "__main__":
    main()