import numpy as np
import torchvision
from data import InferCustom, init_dataloader
import torch

def infer_model(model, test_loader, device):
    """Inference of the model

    Args:
        model: model to infer
        test_loader: dataloader for test data
        device: device used for inference
        subset: just for prettier printing. Defaults to "test".
    """
    model.eval()
    # test_batch_acc = []

    print("Start testing...")
    for X_batch in test_loader:
        logits = model(X_batch.to(device))
        y_pred = logits.max(1)[1].data
        # test_batch_acc.append(np.mean((y_batch.cpu() == y_pred.cpu()).numpy()))
        print(y_pred)
        break

    # test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    # print(f"    {subset} accuracy: {test_accuracy * 100:.2f} %")


def main():
    NUM_CLASSES=6

    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    model = torchvision.models.resnet18(pretrained=False)
    number_feature = model.fc.in_features
    model.fc = torch.nn.Linear(in_features=number_feature , out_features=NUM_CLASSES) 
    model.load_state_dict(torch.load('My_model.pt'))
    model.eval()

    data_dir = "/Users/babkenbrsikyan/Documents/Babken/MLOps/intel_image_cl/data"
    infer_dataset = InferCustom(data_dir + "/seg_pred")
    infer_loader = init_dataloader(infer_dataset, 128)

    infer_model(model, infer_loader, device)

if __name__ == "__main__":
    main()