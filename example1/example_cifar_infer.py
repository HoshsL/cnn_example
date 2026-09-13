
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from model import LeNet
 
def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),  # 首先需将数据集resize成与训练集图像一样的大小
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
 
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
    # 实例化网络
    net = LeNet()
    # 用下述函数载入刚刚训练好的网络模型
    net.load_state_dict(torch.load('Lenet.pth'))
    net.eval()

    # 遍历 img 目录下的图像，并组成一个 batch
    image_dir = Path('img')
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
    )
    if not image_paths:
        raise FileNotFoundError(f'No images found in {image_dir}')

    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            images.append(transform(image.convert('RGB')))
    image_batch = torch.stack(images)  # [N, C, H, W]
 
    with torch.no_grad():
        outputs = net(image_batch)
        predicts = torch.max(outputs, dim=1)[1].numpy()
        # 预测结果也可用softmax，输出十个概率，输出结果中最大概率值对应的索引即为预测标签的索引
        # predict = torch.softmax(outputs, dim=1)
    for image_path, predict in zip(image_paths, predicts):
        print(f'{image_path.name}: {classes[int(predict)]}')
 
 
if __name__ == '__main__':
    main()
