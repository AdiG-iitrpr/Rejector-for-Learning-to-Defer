from dependency import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def metrics_print(net,expert_fn, n_classes, loader):
    '''
    Computes metrics for deferal
    -----
    Arguments:
    net: model
    expert_fn: expert model
    n_classes: number of classes
    loader: data loader
    '''
    correct = 0
    correct_sys = 0
    exp = 0
    exp_total = 0
    total = 0
    real_total = 0
    alone_correct = 0
    batch_counter = 0
    avg_defer = 0
    batch_num_defer = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = outputs.size()[0]  # batch_size
            exp_prediction = expert_fn(images, labels)
            batch_num_defer = 0
            batch_counter += 1
            for i in range(0, batch_size):
                r = (predicted[i].item() == n_classes)
                prediction = predicted[i]
                if predicted[i] == n_classes:
                    max_idx = 0
                    # get second max
                    for j in range(0, n_classes):
                        if outputs.data[i][j] >= outputs.data[i][max_idx]:
                            max_idx = j
                    prediction = max_idx
                else:
                    prediction = predicted[i]
                alone_correct += (prediction == labels[i]).item()
                if r == 0:
                    total += 1
                    correct += (predicted[i] == labels[i]).item()
                    correct_sys += (predicted[i] == labels[i]).item()
                if r == 1:
                    batch_num_defer += 1
                    exp += (exp_prediction[i] == labels[i].item())
                    correct_sys += (exp_prediction[i] == labels[i].item())
                    exp_total += 1
                real_total += 1
            avg_defer += batch_num_defer
            print(batch_num_defer, batch_size)
        print(batch_counter)
        print((avg_defer-batch_num_defer)/(batch_counter-1))
    cov = str(total) + str(" out of") + str(real_total)
    to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                "expert accuracy": 100 * exp / (exp_total + 0.0002),
                "classifier accuracy": 100 * correct / (total + 0.0001),
                "alone classifier": 100 * alone_correct / real_total}
    print(to_print)

def generateHumanLabels(test_loader, expert):
    expert_pred = []
    for i, (input, target) in enumerate(test_loader):
        out = expert.predict(input, target)
        expert_pred.append(out)
    return expert_pred

def getData():
    dataset = 'cifar10'
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
    test_dataset = datasets.__dict__[dataset.upper()]('./data', train=False, download=True,
                                                            transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=128, shuffle=False)
    return test_loader

def main():
    n_dataset = 10 #cifar10   
    model = WideResNet(28, n_dataset + 1, 4, dropRate=0)
    model_type = sys.argv[1]
    model.load_state_dict(torch.load(model_type))
    model = model.to(device)
    test_loader = getData()
    k = int(sys.argv[2])
    expert = synth_expert(k, 10)
    expert_pred = generateHumanLabels(test_loader, expert)
    model.eval()
    metrics_print(model, expert.predict, n_dataset, test_loader)

if __name__ == "__main__":
    main()

# run as python test_defer.py ./baseline_model_200 ./defer_model_200_6_200 6