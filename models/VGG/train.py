import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import vgg
import numpy as np
import warnings


import torch.nn.functional as F
warnings.filterwarnings('ignore')


def generate_label(label, sigma=2, bin_step=1):
    labelset = np.array([i * bin_step for i in range(85)])
    #print(labelset.shape)
    dis = np.exp(-1/2. * np.power((labelset-label)/sigma/sigma, 2))
    dis = dis / dis.sum()
    return dis, labelset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")##优先使用GPU设备
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),##随机裁剪
                                     transforms.RandomHorizontalFlip(),##随机水平翻转
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "FACEA")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:##写入json文件
        json_file.write(json_str)

    batch_size = 16
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "Validation"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images fot validation.".format(train_num,
                                                                           val_num))

    test_data_iter = iter(validate_loader)
    test_image, test_label = test_data_iter.next()
    print(validate_loader)

    model_name = "vgg13"
    net = vgg(model_name=model_name, num_classes=85, init_weights=True)
    net.to(device)
    loss_function =nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.00005)

    epochs =500
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    KL=nn.KLDivLoss(size_average=True,reduce=True)
    for epoch in range(epochs):
        # train
        net.train() ##开启dropout方法
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            images=images.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            #loss = loss_function(outputs, labels)
            labels=labels.cpu().numpy()
            label_tensor=[]
            for i in labels:
                label_dis, _ = generate_label(i)
                label_tensor.append(label_dis)
            label_tensor=torch.Tensor(label_tensor)## data put to cuda memory
            label_tensor = label_tensor.to(device)
            outputs =F.log_softmax(outputs,dim=-1)
            # label_dis, _ = generate_label(labels)
            # print(label_tensor)
            loss=KL(outputs,label_tensor)
            criterion = nn.L1Loss(reduction='mean')
            loss2 = criterion(outputs, label_tensor)
            #loss=loss1+loss2

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)
        # validate
        net.eval()   ##关闭dropout方法
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for data_test in validate_loader:
                val_images, val_labels = data_test
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                predict=torch.softmax(outputs,dim=1)
                # acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                list_tensor = predict.cpu().numpy().tolist()
                pre_list=[]
                for i in range(len(list_tensor)):
                    res=0
                    zz=(np.sum(list_tensor[i]))
                    for index,j in enumerate(list_tensor[i]):
                        p=class_indict[str(index)]
                        res=res+int(class_indict[str(index)])*j
                    pre_list.append((res))
                val_list = val_labels.numpy().tolist()
                error=[]
                for i in range(len(val_list)):
                    error.append(pre_list[i]-val_list[i])
                absError=[]
                for val in error:
                    absError.append(abs(val))
                acc += sum(absError)
        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)
    print('Finished Training')


if __name__ == '__main__':
    main()
