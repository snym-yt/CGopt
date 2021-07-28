# -*- coding: utf-8 -*-

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
    def __init__(self,net,cn,hozon_folder,gakushuuritsu=1e-3,gpu=1):
        self.gakushuuritsu = gakushuuritsu
        self.cn = cn
        self.net = net(cn=cn)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=gakushuuritsu)
        if(gpu):
            # GPUを使う場合
            self.dev = torch.device('cuda')
            self.net.cuda()
        else:
            self.dev = torch.device('cpu')

        self.hozon_folder = hozon_folder
        # 保存のフォルダが元々なければ予め作っておく
        if(not os.path.exists(hozon_folder)):
            os.mkdir(hozon_folder)
        # 最初から開始
        self.mounankai = 0
        self.sonshitsu = []
        self.psnr = []

    def gakushuu(self,dalo_kunren,dalo_kenshou,n_kurikaeshi,n_kaku=5,yokukataru=10):
        print('訓練:%d枚 | 検証%d枚'%(dalo_kunren.len,dalo_kenshou.len))
        t0 = time.time()
        kenshou_data = []
        for da_ken in dalo_kenshou:
            kenshou_data.append(da_ken)
        dalo_kunren.random = True
        print('画像の準備に%.3f分かかった'%((t0-time.time())/60))
        print('==学習開始==')

        t0 = time.time()
        # 何回も繰り返して訓練する
        for kaime in range(self.mounankai,self.mounankai+n_kurikaeshi):
            # ミニバッチ開始
            for i_batch,(x,y) in enumerate(dalo_kunren):
                z = self.net(x.to(self.dev))
                sonshitsu = mse(z,y.to(self.dev)) # 訓練データの損失
                self.opt.zero_grad()
                sonshitsu.backward()
                self.opt.step()

                # 検証データにテスト
                if((i_batch+1)%int(np.ceil(dalo_kunren.nkai/yokukataru))==0 or i_batch==dalo_kunren.nkai-1):
                    self.net.eval()
                    sonshitsu = []
                    psnr = []
                    if(n_kaku):
                        gazou = []
                        n_kaita = 0

                    for x,y in kenshou_data:
                        z = self.net(x.to(self.dev))
                        # 検証データの損失
                        sonshitsu.append(mse(z,y.to(self.dev)).item())
                        # 検証データからできた一部の画像を書く
                        if(n_kaita<n_kaku):
                            x = x.numpy().transpose(0,2,3,1) # 入力
                            y = y.numpy().transpose(0,2,3,1) # 模範
                            z = np.clip(z.cpu().detach().numpy(),0,1).transpose(0,2,3,1) # 出力
                            for i,(xi,yi,zi) in enumerate(zip(x,y,z)):
                                # [入力、出力、模範]
                                gazou.append(np.vstack([xi,zi,yi]))
                                n_kaita += 1
                                if(n_kaita>=n_kaku):
                                    break
                    sonshitsu = np.mean(sonshitsu)
                    psnr.append(10*np.log10((1^2)/sonshitsu))
                    psnr = np.mean(psnr)
                    # if(n_kaku):
                    #     gazou = np.hstack(gazou)
                    #     # imsave(os.path.join(self.hozon_folder,'kekka%03d.jpg'%(kaime+1)),gazou)

                    # 今の状態を出力する
                    print('%d:%d/%d ~ 損失:%.4e %.2f分過ぎた'%(kaime+1,i_batch+1,dalo_kunren.nkai,sonshitsu,(time.time()-t0)/60))
                    print('PSNR = {}'.format(psnr))
                    self.net.train()

            # ミニバッチ一回終了
            self.sonshitsu.append(sonshitsu)
            self.psnr.append(psnr)
            # パラメータや状態を保存する
            # sd = dict(w=self.net.state_dict(),o=self.opt.state_dict(),n=kaime+1,l=self.sonshitsu)
            # torch.save(sd,os.path.join(self.hozon_folder,'netparam.pkl'))

            # 損失（MSE）の変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.gca(ylabel='MSE')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.sonshitsu,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.hozon_folder,'graph.png'))
            plt.close()

            # PSNRの変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.gca(ylabel='PSNR')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.psnr,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.hozon_folder,'psnr.png'))
            plt.close()

    def __call__(self,x,n_batch=8):
        self.net.eval()
        x = torch.Tensor(x)
        y = []
        for i in range(0,len(x),n_batch):
            y.append(self.net(x[i:i+n_batch].to(self.dev)).detach().cpu())
        return torch.cat(y).numpy()

## 実行 ##

kunren_folder = '/content/drive/MyDrive/gausian/dataset/train' # 訓練データのフォルダ
kenshou_folder = '/content/drive/MyDrive/gausian/dataset/test' # 検証データのフォルダ
hozon_folder = 'hozon' # 結果を保存するフォルダ
cn = 1 # チャネル数 (3色データ)
n_batch = 8 # バッチサイズ
px = 128 # 画像の大きさ
n_kurikaeshi = 30 # 何回繰り返すか
n_kaku = 6 # 見るために結果の画像を何枚出力する
yokukataru = 10 # 一回の訓練で何回結果を出力する

# 使うモデルを選ぶ
model = Unet
#net = DnCNN
#net = Win5RB

dalo_kunren = Gazoudalo(kunren_folder,px,n_batch) # 訓練データ
dalo_kenshou = Gazoudalo(kenshou_folder,px,n_batch) # 検証データ
dino = Dinonet(model,cn,hozon_folder)
# 学習開始
dino.gakushuu(dalo_kunren,dalo_kenshou,n_kurikaeshi,n_kaku,yokukataru)