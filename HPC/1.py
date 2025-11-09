# ===== 推理与准确率统计 =====

def inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning inference on: {device}")

    # 加载测试数据
    _, test_loader = get_data_loaders()

    # 加载已保存的最优模型参数
    model = StandardMLP().to(device)
    model.load_state_dict(torch.load("mnist_mlp.pth"))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), -1)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100. * correct / total
    print(f"Inference Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    train_model()
    inference()
