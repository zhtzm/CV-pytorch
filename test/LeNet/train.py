from model.LeNet import LeNet
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import sys
sys.path.append('E:/Github/Image-Classification')


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 定义一个transform用于处理图片
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='E:\Dataset', train=True, 
                                             download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36, 
                                               shuffle=True, num_workers=0)
    
    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='E:\Dataset', train=False, 
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000, 
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    
    net = LeNet().to(device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 5
    save_path = './save_model/Lenet.pth'

    for epoch in range(epochs):
        running_loss = 0.
        for step, data in enumerate(train_loader, start=0):
            inputs, labels = data

            if not inputs.device == device or not labels.device == device:
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    if not val_image.device == device or val_label.device == device:
                        val_image = val_image.to(device=device)
                        val_label = val_label.to(device=device)

                    outputs = net(val_image)
                    predict_y = torch.max(outputs, dim=1)[1]
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    print('epoch:%2d batch:%5d train_loss: %.3f test_accuracy:%.3f' 
                          % (epoch + 1, step + 1, running_loss / 500, accuracy))
                    running_loss = 0.0

    print('Finished Training')

    torch.save(net.state_dict(), save_path)

if __name__ == '__main__':
    main()
