# coding: utf-8
import os
import time
import numpy as np
from sklearn import metrics
import datetime
import sys
import tqdm
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import gc
import matplotlib.pyplot as plt
from torch.autograd import profiler
from thop import profile

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

model_type = "ast"      # ast, hubert, whisper, wav2vec
avec_type = "both"      # both, northwind (scripted), freeform (spontaneous)
adapt_method = "RP"     # LP (Linear probing), RP (reprogramming), FT (Fine-tuning)
prompt_length = 10

FILENAME = f"{model_type}_{avec_type}_{adapt_method}_{prompt_length}s"

print(FILENAME)
print(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class Solver(object):
    def __init__(self, data_loader, valid_loader, test_loader, config):
        gc.collect()
        torch.cuda.empty_cache()
        
        # best
        self.best_train_acc = 0
        self.best_valid_acc = 0
        self.best_train_f1 = 0
        self.best_valid_f1 = 0
        self.best_train_roc_auc = 0
        self.best_valid_roc_auc = 0
        self.best_train_pr_auc = 0
        self.best_valid_pr_auc = 0
        self.best_train_rmse = 9999
        self.best_valid_rmse = 9999
        self.best_test_rmse = 9999
        self.best_train_mae = 9999
        self.best_valid_mae = 9999
        self.best_test_mae = 9999
        self.best_train_pearsonr = -1
        self.best_valid_pearsonr = -1
        self.best_test_pearsonr = -1
        self.dataframe = None
        
        # data loader
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.input_length = config.input_length

        # training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.use_tensorboard = config.use_tensorboard
        self.map_num = config.map_num
        self.pad_num = config.pad_num
        self.prompt_len = config.prompt_len
        self.reprog_front = config.reprog_front
        self.training_mode = config.training_mode
        if self.dataset == "daicwoz":
            self.n_class = 2

        if self.dataset == "edaic":
            self.n_class = 2
            
        if self.dataset == "eatd":
            self.n_class = 2
            
        if self.dataset == "avec2014_both":
            self.n_class = 2
            
        if self.dataset == "avec2014_freeform":
            self.n_class = 2
            
        if self.dataset == "avec2014_northwind":
            self.n_class = 2
            
        # model path and step size
        self.model_save_path = config.model_save_path
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        self.model_type = config.model_type

        # cuda
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.is_cuda = torch.cuda.is_available()
        print(self.is_cuda)
        #print(device)

        # Build model
        self.build_model(self.training_mode)

        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Trainable parameters: {(params)}")
        
        ### Checking
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape}, {param.numel()} parameters")

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f'/data/DAICWOZ/classification_journal/training/confusion_matrix_avec2014/{FILENAME}')

    def get_model(self):
        if self.model_type in ["CNN235.5k"]:
            from models.CNNModel import CNNModel

            return CNNModel(model_type=self.model_type, n_class=self.n_class)#.to(device)
        elif self.model_type == "speechatt":
            from models.SpeechModel import V2SReprogModel

            return V2SReprogModel(
                map_num=self.map_num,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "wav2vec":
            from models.wav2vec import PretrainedWav2VecModel

            return PretrainedWav2VecModel(
                prompt_len=self.prompt_len,
                map_num=self.map_num,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "ast":
            from models.ASTModel import AST
            
            return AST(
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "hubert":
            from models.Hubert import PretrainedHubertModel
            
            return PretrainedHubertModel(
                prompt_len=self.prompt_len,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "hubert_large":
            from models.Hubert_large import PretrainedHubertModel
            
            return PretrainedHubertModel(
                prompt_len=self.prompt_len,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "whisper":
            from models.Whisper import PretrainedWhisperModel
            
            return PretrainedWhisperModel(
                prompt_len=self.prompt_len,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
                is_cuda=self.is_cuda,
            )#.to(device)
            
        elif self.model_type == "whisper_large":
            from models.Whisper_large import PretrainedWhisperModel
            
            return PretrainedWhisperModel(
                prompt_len=self.prompt_len,
                n_class=self.n_class,
                reprog_front=self.reprog_front,
                map_num=self.map_num,
                is_cuda=self.is_cuda,
            )#.to(device)
            

    def build_model(self, training_mode=None):
        # model
        self.model = self.get_model()

        # Fine-tuning: True , Reprogramming: False
        if training_mode == "FT":
            for name, param in self.model.named_parameters():
                # if 'mlp_head' not in name:
                param.requires_grad = True
                
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name}: {param.shape}")
        
        
        # cudaget_acc
        if self.is_cuda:
            self.model.to(device)

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            self.lr, weight_decay=1e-4
        )
        
        # print("Optimizer param groups:")
        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         print(p.shape, p.requires_grad)


    def to_var(self, x):
        if isinstance(x, dict):
            for d in x.keys():
                if torch.cuda.is_available():
                    x[d] = Variable(x[d]).to(device).squeeze()
            return x
        else:
            if torch.cuda.is_available():
                x = x.to(device)
            return Variable(x).squeeze()

    def get_loss_function(self):
        if self.n_class == 1:
            # return nn.MSELoss()
            return nn.HuberLoss(delta=1.0)
            # return nn.BCEWithLogitsLoss()
        elif self.n_class >= 2:
            # return FocalLoss()
            return nn.BCEWithLogitsLoss()
        #return nn.CrossEntropyLoss()

    def train(self):
        # Start training
        start_t = time.time()
        #current_optimizer = "adam"
        current_optimizer = "adam"
        reconst_loss = self.get_loss_function()
        best_metric_valid = 9999
        drop_counter = 0

        # Iterate
        for epoch in range(self.n_epochs):
            gc.collect()
            torch.cuda.empty_cache()
            est_array = []
            gt_array = []
            losses = []
            reconst_loss = self.get_loss_function()
            ctr = 0
            drop_counter += 1
            self.model = self.model.train()
            
            for x, y in self.data_loader:
                gc.collect()
                torch.cuda.empty_cache()
                ctr += 1
                
                for batch in range(self.batch_size):
                    out_train = None
                    y_train = None
                    for i in range(5):
                        X = self.to_var(x[batch][i])
                    # X = self.to_var(x[batch])
                        X = torch.unsqueeze(X,0)
                        Y = self.to_var(y[batch])
                        Y = torch.unsqueeze(Y,0)
                        out = self.model(X)
                        
                        if y_train == None: y_train = Y[0]
                        else: y_train = torch.vstack((y_train, Y[0]))
                        if out_train == None : out_train = out[0]
                        else: out_train = torch.vstack((out_train, out[0]))

                    # y_train = Y
                    # out_train = out[0]
                    
                    ### Checking the memory usages
                    ### context manager for memory profiling
                    # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    #    output = self.model(X)
                    ### print memory usage
                    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
                    
                    
                    # print memory usage summary
                    #print(torch.cuda.memory_summary(2))
                    # 최대 메모리 사용량을 출력합니다.
                    #print(f"Max memory allocated: {torch.cuda.max_memory_allocated(2) / (1024**2):.2f} MB")
                    # 최대 예약된 메모리를 출력합니다.
                    #print(f"Max memory reserved: {torch.cuda.max_memory_reserved(2) / (1024**2):.2f} MB")
                    
                    # FLOPs Calculation 
                    #torch.cuda.set_device(2)  # cuda:2

                    # PROFILING
                    #with profiler.profile(use_cuda=True) as prof:
                    #    flops, params = profile(self.model, inputs=(X,))
                    #    print(f"FLOPs: {flops}, params: {params}")

                    #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                    
                    loss = reconst_loss(out_train.half(), y_train.half())
                    losses.append(float(loss.data))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    """
                    # CALCULATE THE LOSS
                    loss = reconst_loss(out_train.half(), y_train.half())

                    # [1] LOSS VALUE CHECKING
                    print(f"[LOSS] {loss.item()}")

                    # [2] OUTPUT VALUE CHECKING (dead output or not)
                    print(f"[OUTPUT] mean: {out_train.mean().item():.4f}, std: {out_train.std().item():.4f}")

                    # LOSS CHECKING
                    losses.append(float(loss.data))

                    # PARAMETER snapshot save (i.e. tracking the first layer's weight)
                    # old_val = self.model.model.encoder.layers[0].attention.q_proj.weight.data.clone()
                    old_val = self.model.linear.weight.data.clone()
                    
                    # OPTIMIZER INITIALIZATION
                    self.optimizer.zero_grad()

                    # BACKWARD
                    loss.backward()

                    # [3] CHECKING THE GRADIENT
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                print(f"[GRAD] {name}:  grad is None")
                            else:
                                print(f"[GRAD] {name}: mean={param.grad.mean().item():.4e}, max={param.grad.max().item():.4e}")

                    # OPTIMIZER STEP
                    self.optimizer.step()

                    # [4] PARAMETER CHANGES
                    new_val = self.model.model.encoder.layers[0].attention.q_proj.weight.data
                    param_diff = (old_val - new_val).abs().mean().item()
                    print(f"[PARAM CHANGE] Layer0.q_proj.weight diff: {param_diff:.4e}")
                    """
                    
                self.print_log(epoch, ctr, loss, start_t)
                
                
                # estimate
                #print(np.array((sum(out_train)/5).detach().cpu())[0])
                estimated = np.array(out_train.detach().cpu())[0]
                est_array.append(estimated)
                #print(y_train[0].detach().cpu().numpy()[0])
                gt_array.append(y_train.detach().cpu().numpy()[0])
            
            print("\n-----TRAIN-----")
            est_array, gt_array = np.array(est_array), np.array(gt_array)
            loss = np.mean(losses)
            #print(est_array)
            #print(gt_array)
            print("loss: %.4f" % loss)
            # ------------------------
            # Classification (n_class ≥ 2)
            # ------------------------
            if self.n_class >= 2:
                est_array[est_array > 0.5] = 1
                est_array[est_array <= 0.5] = 0

                acc = self.get_acc(est_array, gt_array)
                f1 = self.get_f1(est_array, gt_array)
                roc_auc, pr_auc = self.get_auc(est_array, gt_array)

                if self.best_train_acc < acc: self.best_train_acc = acc
                if self.best_train_f1 < f1: self.best_train_f1 = f1
                if self.best_train_roc_auc < roc_auc: self.best_train_roc_auc = roc_auc
                if self.best_train_pr_auc < pr_auc: self.best_train_pr_auc = pr_auc

                self.writer.add_scalar("Loss/train", loss.item(), epoch)

            # ------------------------
            # Regression (n_class == 1)
            # ------------------------
            elif self.n_class == 1:
                est_flat = est_array.flatten()
                gt_flat = gt_array.flatten()

                from sklearn.metrics import mean_squared_error, mean_absolute_error
                from scipy.stats import pearsonr

                rmse = np.sqrt(mean_squared_error(gt_flat, est_flat))
                mae = mean_absolute_error(gt_flat, est_flat)
                r, _ = pearsonr(gt_flat, est_flat)

                if self.best_train_rmse > rmse: self.best_train_rmse = rmse
                if self.best_train_mae > mae: self.best_train_mae = mae
                if self.best_train_pearsonr < r: self.best_train_pearsonr = r

                print("RMSE: %.4f" % rmse)
                print("MAE : %.4f" % mae)
                print("Pearson r: %.4f" % r)

                self.writer.add_scalar("Loss/train", loss.item(), epoch)
                self.writer.add_scalar("RMSE/train", rmse, epoch)
                self.writer.add_scalar("MAE/train", mae, epoch)
                self.writer.add_scalar("PearsonR/train", r, epoch)
                self.writer.add_scalar("BEST_RMSE/train", self.best_train_rmse, epoch)
                self.writer.add_scalar("BEST_MAE/train", self.best_train_mae, epoch)
                self.writer.add_scalar("BEST_PearsonR/train", self.best_train_pearsonr, epoch)


            print(" ")
            
            # validation
            best_metric_valid = self.validation(best_metric_valid, epoch)

            # schedule optimizer
            current_optimizer, drop_counter = self.opt_schedule(
                current_optimizer, drop_counter
            )

            # -------------------------------
            # Classification: metric print
            # -------------------------------
            if self.n_class >= 2:
                print(f"Best Train ACC      : {self.best_train_acc:.4f}")
                print(f"Best Train F1       : {self.best_train_f1:.4f}")
                print(f"Best Train ROC AUC  : {self.best_train_roc_auc:.4f}")
                print(f"Best Train PR AUC   : {self.best_train_pr_auc:.4f}")
                print(f"Best Valid ACC      : {self.best_valid_acc:.4f}")
                print(f"Best Valid F1       : {self.best_valid_f1:.4f}")
                print(f"Best Valid ROC AUC  : {self.best_valid_roc_auc:.4f}")
                print(f"Best Valid PR AUC   : {self.best_valid_pr_auc:.4f}")

            # -------------------------------
            # Regression: metric print
            # -------------------------------
            elif self.n_class == 1:
                print(f"Best Train RMSE     : {self.best_train_rmse:.4f}")
                print(f"Best Train MAE      : {self.best_train_mae:.4f}")
                print(f"Best Train PearsonR : {self.best_train_pearsonr:.4f}")
                print(f"Best Valid RMSE     : {self.best_valid_rmse:.4f}")
                print(f"Best Valid MAE      : {self.best_valid_mae:.4f}")
                print(f"Best Valid PearsonR : {self.best_valid_pearsonr:.4f}")

            self.test(epoch)

        print(
            "[%s] Train finished. Elapsed: %s"
            % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                datetime.timedelta(seconds=time.time() - start_t),
            )
        )
        
        self.writer.close()
        

    def load(self, filename):
        S = torch.load(filename)
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v for k, v in S.items() if k in model_dict and "delta" not in k
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def opt_schedule(self, current_optimizer, drop_counter):
        # adam to sgd
        if current_optimizer == "adam" and drop_counter == 80:
            self.load(os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth"))
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                0.001,
                momentum=0.9,
                weight_decay=0.0001,
                nesterov=True,
            )
            current_optimizer = "sgd_1"
            drop_counter = 0
            print("sgd 1e-3")
        # first drop
        if current_optimizer == "sgd_1" and drop_counter == 50:
            self.load(os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth"))
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0005
            current_optimizer = "sgd_2"
            drop_counter = 0
            print("sgd 5e-4")

        # second drop
        if current_optimizer == "sgd_2" and drop_counter == 20:
            self.load(os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth"))
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.00001
            current_optimizer = "sgd_3"
            print("sgd 1e-5")

        return current_optimizer, drop_counter

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({"model": model}, filename)

    def get_auc(self, est_array, gt_array):
        roc_aucs = metrics.roc_auc_score(gt_array, est_array, average="macro")
        pr_aucs = metrics.average_precision_score(gt_array,
                                                  est_array,
                                                  average="macro")
        print("roc_auc: %.4f pr_auc: %.4f\n" % (roc_aucs, pr_aucs))
        return roc_aucs, pr_aucs

    def get_acc(self, est_array, gt_array):
        # est_array[est_array>0.5] = 1
        # est_array[est_array<=0.5] = 0
        # print(est_array)
        # print(gt_array)
        acc = metrics.accuracy_score(gt_array, est_array)
        print("Acc: %.4f" % acc)
        return acc

    def get_f1(self, est_array, gt_array):
        f1_weighted = metrics.f1_score(gt_array, est_array, average='weighted')
        print("Weighted F1: %.4f" % f1_weighted)
        f1_macro = metrics.f1_score(gt_array, est_array, average='macro')
        print("Macro F1: %.4f" % f1_macro)
        return f1_weighted
    

    def get_cm(self, est_array, gt_array):
        if self.n_class == 1:
            # Regression: scatter plot + regression metrics print
            est_values = est_array.flatten()
            gt_values = gt_array.flatten()
            
            rmse = np.sqrt(metrics.mean_squared_error(gt_values, est_values))
            mae = metrics.mean_absolute_error(gt_values, est_values)
            r2 = metrics.r2_score(gt_values, est_values)

            print(f"Regression Evaluation:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")

            # 산점도 시각화
            plt.figure(figsize=(5, 5))
            plt.scatter(gt_values, est_values, alpha=0.5, color='royalblue', edgecolor='k')
            plt.plot([0, 63], [0, 63], '--', color='gray')  # 대각선
            plt.xlabel("Ground Truth (BDI-II)")
            plt.ylabel("Prediction")
            plt.title("Regression Prediction vs Ground Truth")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            
            return None  # confusion matrix 없음

        else:
            # Classification confusion matrix
            labels = []
            if self.n_class == 2: labels = ['Normal', 'Depression']
            elif self.n_class == 3: labels = ['Normal', 'Mild', 'Severe']
            elif self.n_class == 6: labels = ['Normal', 'Dep1', 'Dep2', 'Dep3', 'Dep4', 'Dep5']
            
            predict = list(np.argmax(est_array, axis=1))
            groundtruth = list(np.argmax(gt_array, axis=1))
            cm = metrics.confusion_matrix(groundtruth, predict)
            cmd = metrics.ConfusionMatrixDisplay(cm, display_labels=labels)
            return cmd


    def print_log(self, epoch, ctr, loss_bce, start_t):
        if (ctr) % self.log_step == 0:
            log_msg = (
                "[%s] Epoch [%d/%d] Iter [%d/%d] train loss: %.4f Time: %s"
                % (
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    epoch + 1,
                    self.n_epochs,
                    ctr,
                    len(self.data_loader),
                    loss_bce.item(),
                    datetime.timedelta(seconds=time.time() - start_t),
                )
            )
            sys.stdout.write("\r" + log_msg)
            sys.stdout.flush()

        ### if it is the last Iteration (enter the log)
        if ctr == len(self.data_loader):
            sys.stdout.write("\n")

    def validation(self, best_metric, epoch):
        if self.n_class == 1:
            score, loss = self.get_validation_score(epoch)
            if score < best_metric:
                print("best model!\n")
                best_metric = score
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth"),
                )
            return best_metric
        else:
            acc, loss = self.get_validation_score(epoch)
            score = acc
            if score > best_metric:
                print("best model!\n")
                best_metric = score
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth"),
                )
            return best_metric

    def get_validation_score(self, epoch):
        self.model = self.model.eval()
        with torch.no_grad():
            est_array = []
            gt_array = []
            losses = []
            reconst_loss = self.get_loss_function()
            for x, y in tqdm.tqdm(self.valid_loader):
                gc.collect()
                torch.cuda.empty_cache()

                out_valid = None
                y_valid = None
                for i in range(5):
                    X = self.to_var(x[0][i])
                    # X = self.to_var(x[0])
                    X = torch.unsqueeze(X, 0)
                    Y = self.to_var(y)
                    # Y = self.to_var(y[0])
                    Y = torch.unsqueeze(Y, 0)

                    out = self.model(X)
                    if y_valid is None: y_valid = Y[0]
                    else: y_valid = torch.vstack((y_valid, Y[0]))
                    if out_valid is None: out_valid = out[0]
                    else: out_valid = torch.vstack((out_valid, out[0]))

                loss = reconst_loss(out_valid.half(), y_valid.half())
                losses.append(float(loss.data))

                est_array.append(np.array(out_valid.detach().cpu())[0])
                gt_array.append(y_valid.detach().cpu().numpy()[0])

        print("-----VALID-----")
        est_array, gt_array = np.array(est_array), np.array(gt_array)
        loss = np.mean(losses)
        print("loss: %.4f" % loss)

        # --------------------------
        # Regression mode (n_class == 1)
        # --------------------------
        if self.n_class == 1:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from scipy.stats import pearsonr

            est_flat = est_array.flatten()
            gt_flat = gt_array.flatten()

            rmse = np.sqrt(mean_squared_error(gt_flat, est_flat))
            mae = mean_absolute_error(gt_flat, est_flat)
            r, _ = pearsonr(gt_flat, est_flat)

            if self.best_valid_rmse > rmse: self.best_valid_rmse = rmse
            if self.best_valid_mae > mae: self.best_valid_mae = mae
            if self.best_valid_pearsonr < r: self.best_valid_pearsonr = r

            print("RMSE: %.4f" % rmse)
            print("MAE : %.4f" % mae)
            print("Pearson r: %.4f" % r)

            self.writer.add_scalar("Loss/valid", loss, epoch)
            self.writer.add_scalar("RMSE/valid", rmse, epoch)
            self.writer.add_scalar("MAE/valid", mae, epoch)
            self.writer.add_scalar("PearsonR/valid", r, epoch)
            self.writer.add_scalar("BEST_RMSE/valid", self.best_valid_rmse, epoch)
            self.writer.add_scalar("BEST_MAE/valid", self.best_valid_mae, epoch)
            self.writer.add_scalar("BEST_PearsonR/valid", self.best_valid_pearsonr, epoch)


            return rmse, loss  # rmse as validation score

        # --------------------------
        # Classification mode (n_class ≥ 2)
        # --------------------------
        acc = self.get_acc(np.argmax(est_array, axis=1),
                        np.argmax(gt_array, axis=1))
        f1 = self.get_f1(np.argmax(est_array, axis=1),
                        np.argmax(gt_array, axis=1))

        if self.best_valid_acc < acc:
            self.best_valid_acc = acc
            cmd = self.get_cm(est_array, gt_array)
            if cmd is not None:
                cmd.plot()
                plt.savefig(f"./confusion_matrix_avec2014/{FILENAME}_best_valid_acc.png")

        if self.best_valid_f1 < f1:
            self.best_valid_f1 = f1
            cmd = self.get_cm(est_array, gt_array)
            if cmd is not None:
                cmd.plot()
                plt.savefig(f"./confusion_matrix_avec2014/{FILENAME}_best_valid_f1.png")

        roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        if self.best_valid_roc_auc < roc_auc:
            self.best_valid_roc_auc = roc_auc
        if self.best_valid_pr_auc < pr_auc:
            self.best_valid_pr_auc = pr_auc

        self.writer.add_scalar("Loss/valid", loss, epoch)
        self.writer.add_scalar("AUC/ROC", roc_auc, epoch)
        self.writer.add_scalar("AUC/PR", pr_auc, epoch)
        self.writer.add_scalar("ACC", acc, epoch)

        return acc, loss

    def test(self, epoch):
        print("\n-----TEST-----")
        # self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth")), strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        est_array = []
        gt_array = []
        losses = []

        reconst_loss = self.get_loss_function()

        with torch.no_grad():
            for x, y in tqdm.tqdm(self.test_loader):
                gc.collect()
                torch.cuda.empty_cache()

                out_valid = None
                y_valid = None

                for i in range(5):
                    X = self.to_var(x[0][i])   # each chunk
                    X = torch.unsqueeze(X, 0)

                    Y = self.to_var(y)
                    Y = torch.unsqueeze(Y, 0)

                    out = self.model(X)

                    if y_valid is None:
                        y_valid = Y[0]
                    else:
                        y_valid = torch.vstack((y_valid, Y[0]))

                    if out_valid is None:
                        out_valid = out[0]
                    else:
                        out_valid = torch.vstack((out_valid, out[0]))

                loss = reconst_loss(out_valid.half(), y_valid.half())
                losses.append(float(loss.data))

                est_array.append(np.array(out_valid.detach().cpu())[0])
                gt_array.append(y_valid.detach().cpu().numpy()[0])

        est_array = np.array(est_array)
        gt_array = np.array(gt_array)
        loss = np.mean(losses)

        print("loss: %.4f" % loss)

        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr

        est_flat = est_array.flatten()
        gt_flat = gt_array.flatten()

        rmse = np.sqrt(mean_squared_error(gt_flat, est_flat))
        mae = mean_absolute_error(gt_flat, est_flat)
        r, _ = pearsonr(gt_flat, est_flat)

        print(f"Test RMSE        : {rmse:.4f}")
        print(f"Test MAE         : {mae:.4f}")
        print(f"Test Pearson R   : {r:.4f}")

        if self.best_test_rmse > rmse: self.best_test_rmse = rmse
        if self.best_test_mae > mae: self.best_test_mae = mae
        if self.best_test_pearsonr < r: self.best_test_pearsonr = r

        self.writer.add_scalar("Loss/test", loss, epoch)
        self.writer.add_scalar("RMSE/test", rmse, epoch)
        self.writer.add_scalar("MAE/test", mae, epoch)
        self.writer.add_scalar("PearsonR/test", r, epoch)
        self.writer.add_scalar("BEST_RMSE/test", self.best_test_rmse, epoch)
        self.writer.add_scalar("BEST_MAE/test", self.best_test_mae, epoch)
        self.writer.add_scalar("BEST_PearsonR/test", self.best_test_pearsonr, epoch)

        return rmse, mae, r


    def final_test(self):
        print("\n-----FINAL TEST-----")
        self.model.load_state_dict(torch.load(os.path.join(self.model_save_path, f"best_model_{FILENAME}.pth")), strict=False)
        self.model = self.model.to(device)
        self.model.eval()

        est_array = []
        gt_array = []
        losses = []

        reconst_loss = self.get_loss_function()

        with torch.no_grad():
            for x, y in tqdm.tqdm(self.test_loader):
                gc.collect()
                torch.cuda.empty_cache()

                out_valid = None
                y_valid = None

                for i in range(5):
                    X = self.to_var(x[0][i])  # each chunk
                    X = torch.unsqueeze(X, 0)

                    Y = self.to_var(y)
                    Y = torch.unsqueeze(Y, 0)

                    out = self.model(X)

                    if y_valid is None:
                        y_valid = Y[0]
                    else:
                        y_valid = torch.vstack((y_valid, Y[0]))

                    if out_valid is None:
                        out_valid = out[0]
                    else:
                        out_valid = torch.vstack((out_valid, out[0]))

                loss = reconst_loss(out_valid.half(), y_valid.half())
                losses.append(float(loss.data))

                est_array.append(np.array(out_valid.detach().cpu())[0])
                gt_array.append(y_valid.detach().cpu().numpy()[0])

        est_array = np.array(est_array)
        gt_array = np.array(gt_array)
        loss = np.mean(losses)

        print("loss: %.4f" % loss)

        # Regression metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        from scipy.stats import pearsonr

        est_flat = est_array.flatten()
        gt_flat = gt_array.flatten()

        rmse = np.sqrt(mean_squared_error(gt_flat, est_flat))
        mae = mean_absolute_error(gt_flat, est_flat)
        r, _ = pearsonr(gt_flat, est_flat)

        print(f"FIANL Test RMSE        : {rmse:.4f}")
        print(f"FIANL Test MAE         : {mae:.4f}")
        print(f"FIANL Test Pearson R   : {r:.4f}")
        
        return rmse, mae, r
