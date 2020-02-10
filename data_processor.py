
import torch
from torchvision import datasets, transforms


train_set = datasets.FashionMNIST(root='./data/', train=True, download=True, 
	transform=transforms.Compose([
            transforms.ToTensor()
            ]))

test_set = datasets.FashionMNIST(root='./data/', train=False, download=True, 
	transform=transforms.Compose([
            transforms.ToTensor()
            ]))

def main():
	pass 
	
if __name__ == "__main__":

	main()