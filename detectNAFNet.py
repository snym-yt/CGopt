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
class Block(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_channels)
        self.conv = nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding='same')
        self.dconv = nn.Conv2d(in_channels, )
        self.simple_gate =
        self.sca = 

    def forward(self, x):
        y = x
        x = self.layer_norm(x)
        x = self.conv(x)
        x = self.dconv(x)
        x = self.simple_gate(x)
        x = self.sca(x)
        x = self.conv(x)
        x = torch.add(x,y)
        y = x
        x = self.layer_norm(x)
        x = self.conv(x)
        x = self.simple_gate(x)
        x = self.conv(x)
        x = torch.add(x,y)

        return x

class NAFNet(nn.Module):
    def __init__(self) -> None:
        super(NAFNet, self).__init__()
        self.B1 = Block(3, 64, 64)

        self.maxpool = nn.MaxPool2d(2)


    def forward(self, x):
        x = Block()


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
    def __init__(self,net,cn,save_folder,learning_rate=1e-3,gpu=0):
        self.learning_rate = learning_rate
        self.cn = cn
        self.net = net(cn=cn)
        self.opt = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
        if(gpu):
            # GPUを使う場合
            self.dev = torch.device('cuda')
            self.net.cuda()
        else:
            self.dev = torch.device('cpu')

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
                loss = mse(z,y.to(self.dev)) # 訓練データの損失
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 検証データにテスト
                if((i_batch+1)%int(np.ceil(dalo_train.nkai/cnt_multi))==0 or i_batch==dalo_train.nkai-1):
                    self.net.eval()
                    loss = []
                    psnr = []
                    if(n_kaku):
                        gazou = []
                        n_kaita = 0

                    for x,y in test_data:
                        z = self.net(x.to(self.dev))
                        # 検証データの損失
                        loss.append(mse(z,y.to(self.dev)).item())
                        psnr.append(10*np.log10((1^2)/mse(z,y.to(self.dev)).item()))
                        # 検証データからできた一部の画像を書く
                        # if(n_kaita<n_kaku):
                        #     x = x.numpy().transpose(0,2,3,1) # 入力
                        #     y = y.numpy().transpose(0,2,3,1) # 模範
                        #     z = np.clip(z.cpu().detach().numpy(),0,1).transpose(0,2,3,1) # 出力
                        #     for i,(xi,yi,zi) in enumerate(zip(x,y,z)):
                        #         # [入力、出力、模範]
                        #         gazou.append(np.vstack([xi,zi,yi]))
                        #         n_kaita += 1
                        #         if(n_kaita>=n_kaku):
                        #             break
                    loss = np.mean(loss)
                    psnr = np.mean(psnr)
                    # if(n_kaku):
                    #     gazou = np.hstack(gazou)
                    #     # imsave(os.path.join(self.save_folder,'kekka%03d.jpg'%(kaime+1)),gazou)

                    # 今の状態を出力する
                    print('%d:%d/%d ~ 損失:%.4e %.2f分過ぎた'%(kaime+1,i_batch+1,dalo_train.nkai,loss,(time.time()-t0)/60))
                    print('PSNR = {}'.format(psnr))
                    self.net.train()

            # ミニバッチ一回終了
            self.loss.append(loss)
            self.psnr.append(psnr)
            # パラメータや状態を保存する
            # sd = dict(w=self.net.state_dict(),o=self.opt.state_dict(),n=kaime+1,l=self.loss)
            # torch.save(sd,os.path.join(self.save_folder,'netparam.pkl'))

            # 損失（MSE）の変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('MSE')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.loss,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'graph_MSE.png'))
            plt.close()

            # PSNRの変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('PSNR(MSE)')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.psnr,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'psnr_MSE.png'))
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
n_loop = 30 # 何回繰り返すか
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
