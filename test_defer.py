from dependency import *
from baseline_model import WideResNet
from defer_model import DeferModel
from expert import synth_expert

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
    model = WideResNet(28, n_dataset, 4, dropRate=0)
    baseline_model_type = sys.argv[1]
    defer_model_type = sys.argv[2]
    model.load_state_dict(torch.load(baseline_model_type))
    model = model.to(device)
    model_defer = DeferModel()
    model_defer.load_state_dict(torch.load(defer_model_type))
    model_defer = model_defer.to(device)
    test_loader = getData()
    k = int(sys.argv[3])
    expert = synth_expert(k, 10)
    expert_pred = generateHumanLabels(test_loader, expert)
    model.eval()
    model_defer.eval()
    acc = 0.0
    def_acc = 0.0
    for i, (input, target) in enumerate(test_loader):
        corr = 0.0
        defer = 0.0
        target = target.to(device)
        input = input.to(device)
        machine_output = model(input)
        machine_pred_class = torch.argmax(machine_output, dim=1)
        defer_output = model_defer(machine_output)
        defer_output = defer_output.squeeze()
        defer_output = torch.round(torch.sigmoid(defer_output))
        for j in range(len(defer_output)):
            if defer_output[j]==0:
                if machine_pred_class[j]==target[j]:
                    corr+=1
            else:
                defer += 1
                if expert_pred[i][j]==target[j]:
                    corr+=1
        percent = ((corr*100)/len(defer_output))
        def_acc += defer
        acc += percent
        print(acc/(i+1),"\t", def_acc/(i+1))

if __name__ == "__main__":
    main()

# run as python test_defer.py ./baseline_model_200 ./defer_model_200_6_200 6