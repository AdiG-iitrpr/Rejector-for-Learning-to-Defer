class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Utility(object):
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


    def metrics_print(self, net, expert_fn, n_classes, loader):
        '''
        Computes metrics for deferal
        -----
        Arguments:
        net: model
        expert_fn: expert model predict function
        n_classes: number of classes
        loader: data loader
        '''
        correct = 0
        correct_sys = 0
        exp = 0
        exp_total = 0
        total = 0
        real_total = 0
        with torch.no_grad():
            for data in loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                batch_size = outputs.size()[0]  # batch_size
                exp_prediction = expert_fn(images, labels)
                for i in range(0, batch_size):
                    r = (predicted[i].item() == n_classes)
                    if r == 0:
                        total += 1
                        correct += (predicted[i] == labels[i]).item()
                        correct_sys += (predicted[i] == labels[i]).item()
                    if r == 1:
                        exp += (exp_prediction[i] == labels[i].item())
                        correct_sys += (exp_prediction[i] == labels[i].item())
                        exp_total += 1
                    real_total += 1
        cov = str(total) + str(" out of") + str(real_total)
        to_print = {"coverage": cov, "system accuracy": 100 * correct_sys / real_total,
                    "expert accuracy": 100 * exp / (exp_total + 0.0002),
                    "classifier accuracy": 100 * correct / (total + 0.0001)}
        print(to_print)
