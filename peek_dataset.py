from datasets import load_dataset

def peek_dataset():
    dataset = load_dataset("deccan-ai/insuranceQA-v2")
    print("Splits:", dataset.keys())
    for split in dataset.keys():
        print(f"\nPeek into {split}:")
        print(dataset[split][0])

if __name__ == "__main__":
    peek_dataset()
