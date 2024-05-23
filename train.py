import os
import time
import torch
import argparse
import dill
from utilits_lib.diffusion_model import diffusion_model
#parse
def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--DataPath", type=str, help="Input processed data.", default="./dataProcessed/mnist_dataset.pkl")
    parse.add_argument("--ImgSize", type=int, help="Image size.", default=32)
    parse.add_argument("--TimeSteps", type=int, help="Steps that original images be broken.", default=300)
    parse.add_argument("--LearningRate", type=float, help="Learning rate.", default=1e-3)
    parse.add_argument("--Epochs", type=int, help="epoch.", default=3000)
    parse.add_argument("--BatchSize", type=int, help="Batch size.", default=32)
    parse.add_argument("--Model_name", type=str, help="Save the generator with this name.")
    args = parse.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    with open(args.DataPath, 'rb') as file:
        dataset = dill.load(file)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.time()
    dm = diffusion_model(T=args.TimeSteps, img_size=args.ImgSize, batch_size=args.BatchSize)
    dm.train(dataset, lr = args.LearningRate, epochs=args.Epochs)
    print("time:", (time.time() - start)/60, "mins")

    if not os.path.exists("./modelSaving/"):
        os.makedirs("./modelSaving/")

    torch.save(dm, f"./modelSaving/{args.Model_name}.pt")