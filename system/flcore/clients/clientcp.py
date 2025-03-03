import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
import os


class clientCP:
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        self.model = copy.deepcopy(args.model)
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.dp=args.difference_privacy
        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps
        self.noise= {}
        self.grad_records = []
        result_dir = "gradient_log"
        os.makedirs(result_dir, exist_ok=True)
        if args.difference_privacy:
            filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}_{args.difference_privacy_layer}.txt"
        else:
            filename = f"gradient_log_client{self.id}_{args.dataset}_{args.global_rounds}_{args.local_learning_rate:.4f}.txt"
        self.filepath = os.path.join(result_dir, filename)



        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.lamda = args.lamda

        in_dim = list(args.model.head.parameters())[0].shape[1]
        self.context = torch.rand(1, in_dim).to(self.device)

        self.model = Ensemble(
            model=self.model,  # model is the global model
            cs=copy.deepcopy(kwargs['ConditionalSelection']),
            head_g=copy.deepcopy(self.model.head),  # head is the global head
            feature_extractor=copy.deepcopy(self.model.feature_extractor)
            # feature_extractor is the global feature_extractor
        )
        # 假设 args.dpm = "head"
        attr_path = f"model.{args.difference_privacy_layer}".split(".")  # 拼接路径并分割为列表
        # 逐层获取属性
        target_module = self
        for attr in attr_path:
            target_module = getattr(target_module, attr)  # 动态获取模块

        # 获取目标模块的 named_parameters()
        self.dp_layer = target_module

        self.opt = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.pm_train = []
        self.pm_test = []

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def set_parameters(self, feature_extractor):
        for new_param, old_param in zip(feature_extractor.parameters(),
                                        self.model.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

        for new_param, old_param in zip(feature_extractor.parameters(), self.model.feature_extractor.parameters()):
            old_param.data = new_param.data.clone()

    # set the head of the global model to the local model
    def set_head_g(self, head):
        headw_ps = []
        for name, mat in self.model.model.head.named_parameters():
            if 'weight' in name:
                headw_ps.append(mat.data)
        headw_p = headw_ps[-1]
        for mat in headw_ps[-2::-1]:
            headw_p = torch.matmul(headw_p, mat)
        headw_p.detach_()
        self.context = torch.sum(headw_p, dim=0, keepdim=True)

        for new_param, old_param in zip(head.parameters(), self.model.head_g.parameters()):
            old_param.data = new_param.data.clone()

    def set_cs(self, cs):
        for new_param, old_param in zip(cs.parameters(), self.model.gate.cs.parameters()):
            old_param.data = new_param.data.clone()

    def save_con_items(self, items, tag='', item_path=None):
        self.save_item(self.pm_train, 'pm_train' + '_' + tag, item_path)
        self.save_item(self.pm_test, 'pm_test' + '_' + tag, item_path)
        for idx, it in enumerate(items):
            self.save_item(it, 'item_' + str(idx) + '_' + tag, item_path)

    def generate_upload_head(self):
        for (np, pp), (ng, pg) in zip(self.model.model.head.named_parameters(), self.model.head_g.named_parameters()):
            pg.data = pp*0.5+pg*0.5

    def get_module_grad_norm(self,model) -> float:
        """
        计算给定模块的所有参数梯度的 L2 范数并返回。
        如果某个参数没有梯度（grad=None），则跳过。
        """
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        return total_norm_sq ** 0.5
    def test_metrics(self):
        testloader = self.load_test_data()
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        self.model.gate.pm_ = []
        self.model.gate.gm_ = []
        self.pm_test = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x, is_rep=False, context=self.context)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        self.pm_test.extend(self.model.gate.pm_)

        return test_acc, test_num, auc

    def train_cs_model(self, round, args):
        # if round > 0 and self.dp:
        #     for name, param in self.dp_layer.named_parameters():
        #         param.data =self.noise[name]

        # if round > 100:
        #     for param in self.model.head_g.parameters():
        #         param.requires_grad = True
        #     for param in self.model.feature_extractor.parameters():
        #         param.requires_grad = True
        #     for param in self.model.gate.cs.parameters():
        #         param.requires_grad = False

        initial_params = {name: param.clone().detach() for name, param in self.dp_layer.named_parameters()}

        trainloader = self.load_train_data()
        self.model.train()
        for _ in range(self.local_steps):
            self.model.gate.pm = []
            self.model.gate.gm = []
            self.pm_train = []
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output, rep, rep_base = self.model(x, is_rep=True, context=self.context)

                loss = self.loss(output, y)
                loss += MMD(rep, rep_base, 'rbf', self.device) * self.lamda
                self.opt.zero_grad()
                loss.backward()

                self.opt.step()
        grad_norm_head = self.get_module_grad_norm(self.model.model.head)  # global model
        grad_norm_feat = self.get_module_grad_norm(self.model.model.feature_extractor)

        # 你也可以再加一个 "整体" 的 grad_norm
        grad_norm_ensemble = self.get_module_grad_norm(self.model)

        # 接下来你可以将这些范数记录到日志、保存到文件、
        # 或者可视化，比如 print/log/wandb/tensorboard 等
        record_dict = {
            "round": round,
            "grad_norm_head": grad_norm_head,
            "grad_norm_feat": grad_norm_feat,
            # "grad_norm_cs": grad_norm_cs,
            "grad_norm_ensemble": grad_norm_ensemble
        }
        with open(self.filepath, "a") as f:
            f.write(str(record_dict) + "\n")
        # Clip and add noise for DP if enabled
        clip_value = 0.002
        epsilon = 0.5
        delta = 1e-5

        if self.dp:
            param_diff = {}
            diff_norms = []

            for name, param in self.dp_layer.named_parameters():
                param_diff[name] = (param - initial_params[name]).detach()
                diff_norm = param_diff[name].norm(p=2).item()
                diff_norms.append(diff_norm)


            for name, diff in param_diff.items():
                norm = torch.norm(diff)
                if norm > clip_value:
                    diff = diff / norm * clip_value

                noise_std = (clip_value / epsilon) * torch.sqrt(
                    torch.tensor(2.0) * torch.log(torch.tensor(1.25 / delta)))
                noise = torch.normal(mean=0, std=noise_std, size=diff.shape).to(diff.device)
                self.noise[name] = noise
                param_diff[name] += noise

                for name, param in self.dp_layer.named_parameters():
                    param.data = initial_params[name] + param_diff[name]

        self.pm_train.extend(self.model.gate.pm)
        scores = [torch.mean(pm).item() for pm in self.pm_train]
        print(np.mean(scores), np.std(scores))

        # Save model at 100th round
        if round == 999:
            import os
            save_dir = "pretrain"
            os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
            if self.dp:
                filename = f"results_{args.dataset}_client{self.id}_{args.global_rounds}_{args.local_learning_rate:.4f}_{args.difference_privacy_layer}.pt"
            else:
                filename = f"results_{args.dataset}_client{self.id}_{args.global_rounds}_{args.local_learning_rate:.4f}.pt"
            save_path = os.path.join(save_dir, filename)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")



def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


class Ensemble(nn.Module):
    def __init__(self, model, cs, head_g, feature_extractor) -> None:
        super().__init__()

        self.model = model
        self.head_g = head_g  # head_g is the global head
        self.feature_extractor = feature_extractor # feature_extractor is the global feature_extractor

        for param in self.head_g.parameters():
            param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.flag = 0
        self.tau = 1
        self.hard = False
        self.context = None

        self.gate = Gate(cs)

    def forward(self, x, is_rep=False, context=None):

        rep = self.model.feature_extractor(x)  # feature_extractor is the global feature_extractor
        gate_in = rep

        if context != None:
            context = F.normalize(context, p=2, dim=1)
            if type(x) == type([]):
                self.context = torch.tile(context, (x[0].shape[0], 1))
            else:
                self.context = torch.tile(context, (x.shape[0], 1))

        if self.context != None:
            gate_in = rep * self.context
        if self.flag == 0:
            rep_p, rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p) + self.head_g(rep_g)
        elif self.flag == 1:
            rep_p = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.model.head(rep_p)
        else:
            rep_g = self.gate(rep, self.tau, self.hard, gate_in, self.flag)
            output = self.head_g(rep_g)

        if is_rep:
            return output, rep, self.feature_extractor(x)
        else:
            return output


class Gate(nn.Module):
    def __init__(self, cs) -> None:
        super().__init__()

        self.cs = cs
        self.pm = []
        self.gm = []
        self.pm_ = []
        self.gm_ = []

    def forward(self, rep, tau=1, hard=False, context=None, flag=0):
        pm, gm = self.cs(context, tau=tau, hard=hard)
        if self.training:
            self.pm.extend(pm)
            self.gm.extend(gm)
        else:
            self.pm_.extend(pm)
            self.gm_.extend(gm)

        if flag == 0:
            rep_p = rep * pm
            rep_g = rep * gm
            return rep_p, rep_g
        elif flag == 1:
            return rep * pm
        else:
            return rep * gm