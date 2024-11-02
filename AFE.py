import logging
import parser
import torch
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def calculate_energy_accuracy(args, i, model, test_loader, device, T=1):
    losses, accs = 0.0, 0.0
    criterion = torch.nn.CrossEntropyLoss()
    energies = []
    model.eval()

    with torch.no_grad():

        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            ## accuracy, loss
            acc = accuracy(output, target)
            accs += acc[0].item() * target.size(0)
            losses += loss.item() * target.size(0)
            ## energy
            energy = -T * (torch.logsumexp(output / T, dim=1))
            energies.append(energy.detach().cpu())

        ## stack all energy
        energies = torch.cat(energies, 0)
        # avg energy-based meta-distribution, avg accuracy, avg loss
        avg_energies = 0
        if args.score == 'EMD':
            avg_energies = torch.log_softmax(energies, dim=0).mean()
            avg_energies = torch.log(-avg_energies).item()
        elif args.score == 'EMD1':
            avg_energies = torch.log_softmax(energies, dim=0).mean()
            avg_energies = -avg_energies.item()
        elif args.score == 'AVG':
            avg_energies = energies.mean()
            avg_energies = -avg_energies.item()
        avg_accs = accs / len(test_loader.dataset)
        avg_losses = losses / len(test_loader.dataset)

    logging.info(f"%s Dataset%d: Test Energy: %.2f, Test ACC: %.2f, Test loss: %.2f" \
                 % (args.dataset, i, avg_energies, avg_accs, avg_losses))
    return avg_energies, avg_accs


def main():
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model here
    model = None
    model.to(device)
    model.eval()
    # load dataset here
    testloaders = None

    energies, accuracies = [], []
    for i, testloader in enumerate(testloaders[-1]):
        energy, acc = calculate_energy_accuracy(args, i, model, testloader, device, args.T)
        energies.append(energy)
        accuracies.append(acc)
    energies, accuracies = np.array(energies), np.array(accuracies)

    rho, pval = stats.spearmanr(energies, accuracies)
    logging.info(f'Spearman\'s rank correlation-rho %.3f' % (rho))
    logging.info(f'Spearman\'s rank correlation-pval %.3f' % (pval))
    rho, pval = stats.pearsonr(energies, accuracies)
    logging.info(f'Pearsons correlation-rho %.3f' % (rho))
    logging.info(f'Pearsons correlation-pval %.3f' % (pval))
    rho, pval = stats.kendalltau(energies, accuracies)
    logging.info(f'Kendall\'s rank correlation-rho %.3f' % (rho))
    logging.info(f'Kendall\'s rank correlation-pval %.3f' % (pval))

    slr = LinearRegression()
    slr.fit(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    R2 = slr.score(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    logging.info(f'Linear coefficient of determination-R2 %.3f' % (R2))

    robust_reg = HuberRegressor()
    robust_reg.fit(energies.reshape(-1, 1), accuracies.reshape(-1))
    robust_R2 = robust_reg.score(energies.reshape(-1, 1), accuracies.reshape(-1, 1))
    logging.info(f'Robust linear coefficient of determination-robust_R2 %.3f\n' % (robust_R2))

    logging.info(f"==> Evaluating on unseen test sets of {args.dataset}...")
    for i, testloader in enumerate(testloaders[:-1]):
        test_energy, test_acc = calculate_energy_accuracy(args, i, model, testloader, device, args.T_MAE)
        test_energy, test_acc = np.array(test_energy), np.array(test_acc)

        pred = slr.predict(test_energy.reshape(-1, 1))
        MAE = mean_absolute_error(pred, test_acc.reshape(-1, 1))
        logging.info(
            'Linear regressor: %s Unseen Testset%d: True Energy %.2f, True Acc: %.2f, Pred Acc: %.2f, MAE: %.2f' \
            % (args.dataset, i, test_energy, test_acc, pred, MAE))


if __name__ == "__main__":
    main()
