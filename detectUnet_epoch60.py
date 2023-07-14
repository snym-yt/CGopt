# -*- coding: utf-8 -*-
from os import path
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave,imread
from skimage.transform import resize
import os,time,torch
from torch import nn
mse = nn.MSELoss()




## ニューラルネットワークモデル ##

class Unet(nn.Module):
    def __init__(self,cn=3):
        super(Unet,self).__init__()

        self.copu1 = nn.Sequential(
            nn.Conv2d(cn,48,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48,48,3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        for i in range(2,6):
            self.add_module('copu%d'%i,
                nn.Sequential(
                    nn.Conv2d(48,48,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)
                )
            )

        self.coasa1 = nn.Sequential(
            nn.Conv2d(48,48,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48,48,3,stride=2,padding=1,output_padding=1)
        )

        self.coasa2 = nn.Sequential(
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96,96,3,stride=2,padding=1,output_padding=1)
        )

        for i in range(3,6):
            self.add_module('coasa%d'%i,
                nn.Sequential(
                    nn.Conv2d(144,96,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(96,96,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(96,96,3,stride=2,padding=1,output_padding=1)
                )
            )

        self.coli = nn.Sequential(
            nn.Conv2d(96+cn,64,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,32,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,cn,3,stride=1,padding=1),
            nn.LeakyReLU(0.1)
        )

        for l in self.modules(): # 重みの初期値
            if(type(l) in (nn.ConvTranspose2d,nn.Conv2d)):
                nn.init.kaiming_normal_(l.weight.data)
                l.bias.data.zero_()

    def forward(self,x):
        x1 = self.copu1(x)
        x2 = self.copu2(x1)
        x3 = self.copu3(x2)
        x4 = self.copu4(x3)
        x5 = self.copu5(x4)

        z = self.coasa1(x5)
        z = self.coasa2(torch.cat((z,x4),1))
        z = self.coasa3(torch.cat((z,x3),1))
        z = self.coasa4(torch.cat((z,x2),1))
        z = self.coasa5(torch.cat((z,x1),1))

        return self.coli(torch.cat((z,x),1))


## データローダ ##

class Gazoudalo:
    def __init__(self,folder,px,n_batch=4,random=False):
        self.input_folder = folder + '/input' # データの保存したinputフォルダ
        self.teach_folder = folder + '/teach' # データの保存したteachフォルダ
        self.px = px # 画像の大きさ
        self.n_batch = n_batch # バッチサイズ
        self.random = random # ランダムするかどうか
        self.input_file = np.sort(glob(os.path.join(self.input_folder,'*.jpg'))) # フォルダの中の全部の画像の名前
        self.teach_file = np.sort(glob(os.path.join(self.teach_folder,'*.jpg'))) # フォルダの中の全部の画像の名前
        self.len = len(self.input_file) # 画像の枚数
        self.nkai = int(np.ceil(self.len/n_batch)) # このバッチサイズで繰り返すことができる回数

    def __iter__(self):
        self.i_iter = 0
        if(self.random):
            np.random.seed(None)
            # ランダムで順番を入れ替える
            self.i_rand = np.random.permutation(self.len)
        else:
            np.random.seed(0)
            self.i_rand = np.arange(self.len)
        return self

    def __next__(self):
        if(self.i_iter>=self.len):
            raise StopIteration

        x = []
        y = []
        for i in self.i_rand[self.i_iter:self.i_iter+self.n_batch]:
            xi = imread(self.input_file[i])/255. # input読み込み
            yi = imread(self.teach_file[i])/255. # teach読み込み

            # 指定のサイズに変える
            xi = resize(xi,(self.px,self.px),anti_aliasing=True,mode='constant')
            yi = resize(yi,(self.px,self.px),anti_aliasing=True,mode='constant')
            xi = torch.Tensor(xi[None, :, :])
            yi = torch.Tensor(yi[None, :, :])
            x.append(xi)
            y.append(yi)

        x = torch.stack(x)
        y = torch.stack(y)
        self.i_iter += self.n_batch
        return x,y


## 画像変換器 ##

class Dinonet:
    def __init__(self,net,cn,save_folder,learning_rate=1e-3):
        self.learning_rate = learning_rate
        self.cn = cn
        self.net = net(cn=cn)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        self.dev = torch.device("cuda" if (torch.cuda.device_count()>0) else "cpu")
        self.net.to(self.dev)

        self.save_folder = save_folder
        # 保存のフォルダが元々なければ予め作っておく
        if(not os.path.exists(save_folder)):
            os.mkdir(save_folder)
        # 最初から開始
        self.mounankai = 0
        self.loss = []
        self.psnr = []

    def gakushuu(self,dalo_train,dalo_test,n_loop,n_kaku=5,cnt_multi=10):
        print('訓練:%d枚 | 検証%d枚'%(dalo_train.len,dalo_test.len))
        t0 = time.time()
        test_data = []
        for da_ken in dalo_test:
            test_data.append(da_ken)
        dalo_train.random = True
        print('画像の準備に%.3f分かかった'%((time.time() - t0)/60))
        print('==学習開始==')

        t0 = time.time()
        # 何回も繰り返して訓練する
        for kaime in range(self.mounankai,self.mounankai+n_loop):
            # ミニバッチ開始
            for i_batch,(x,y) in enumerate(dalo_train):
                z = self.net(x.to(self.dev))
                lossMSE = mse(z,y.to(self.dev)) # 訓練データの損失
                self.opt.zero_grad()
                lossMSE.backward()
                self.opt.step()

                # 検証データにテスト
                if((i_batch+1)%int(np.ceil(dalo_train.nkai/cnt_multi))==0 or i_batch==dalo_train.nkai-1):
                    self.net.eval()
                    lossMSE = []
                    psnrMSE = []

                    for x,y in test_data:
                        z = self.net(x.to(self.dev))
                        # 検証データの損失
                        lossMSE.append(mse(z,y.to(self.dev)).item())
                        psnrMSE.append(10*np.log10((1**2)/mse(z,y.to(self.dev)).item()))

                        plt.figure(figsize=[5,4])
                        plt.imshow(z[0].squeeze().to('cpu').detach().numpy(), cmap='gray')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_folder,'denoised_image_Unet_epoch60.png'))
                        plt.close()


                    lossMSE = np.mean(lossMSE)
                    psnrMSE = np.mean(psnrMSE)

                    # 今の状態を出力する
                    print('%d:%d/%d ~ 損失:%.4e %.2f分過ぎた'%(kaime+1,i_batch+1,dalo_train.nkai,lossMSE,(time.time()-t0)/60))
                    print('PSNR = {}'.format(psnrMSE))
                    self.net.train()

            # ミニバッチ一回終了
            self.loss.append(lossMSE)
            self.psnr.append(psnrMSE)

            # 損失（MSE）の変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('MSE(U-net: batch size = 8)')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.loss,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'loss_Unet_epoch60.png'))
            plt.close()

            # PSNRの変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('PSNR(U-net: batch size = 8) [dB]')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.psnr,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'psnr_Unet_epoch60.png'))
            plt.close()

    def __call__(self,x,n_batch=8):
        self.net.eval()
        x = torch.Tensor(x)
        y = []
        for i in range(0,len(x),n_batch):
            y.append(self.net(x[i:i+n_batch].to(self.dev)).detach().cpu())
        return torch.cat(y).numpy()

## 実行 ##

train_folder = path.join(path.dirname(__file__), 'train') # 訓練データのフォルダ
test_folder = path.join(path.dirname(__file__), 'test') # 検証データのフォルダ
save_folder = path.join(path.dirname(__file__), 'save') # 結果を保存するフォルダ
cn = 1 # チャネル数 (3色データ)
n_batch = 8 # バッチサイズ
px = 128 # 画像の大きさ
n_loop = 60 # 何回繰り返すか
n_kaku = 6 # 見るために結果の画像を何枚出力する
cnt_multi = 10 # 一回の訓練で何回結果を出力する

# 使うモデルを選ぶ
model = Unet


dalo_train = Gazoudalo(train_folder,px,n_batch) # 訓練データ
dalo_test = Gazoudalo(test_folder,px,n_batch) # 検証データ
dino = Dinonet(model,cn,save_folder)
# 学習開始
dino.gakushuu(dalo_train,dalo_test,n_loop,n_kaku,cnt_multi)
print("finish")
