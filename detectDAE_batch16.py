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
    
class autoencoder(nn.Module):
    def __init__(self, cn=3):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128 * 128, 256),  # 入力サイズあってる？
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(True),
            nn.Linear(256, 128 * 128),  # 出力サイズも
            nn.Sigmoid())

    def forward(self, x):
        # print("Hello World")
        x = x.view(x.size(0),-1)  # 入力を1次元に変換
        # print(x.size())
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(x.size(0),1, 128,128)  # 出力を画像の形状に変換
        return x





## データローダ ##

class Gazoudalo:
    def __init__(self,folder,px,BATCH_SIZE=4,random=False):
        self.input_folder = folder + '/input' # データの保存したinputフォルダ
        self.teach_folder = folder + '/teach' # データの保存したteachフォルダ
        self.px = px # 画像の大きさ
        self.BATCH_SIZE = BATCH_SIZE # バッチサイズ
        self.random = random # ランダムするかどうか
        self.input_file = np.sort(glob(os.path.join(self.input_folder,'*.jpg'))) # フォルダの中の全部の画像の名前
        self.teach_file = np.sort(glob(os.path.join(self.teach_folder,'*.jpg'))) # フォルダの中の全部の画像の名前
        self.len = len(self.input_file) # 画像の枚数
        self.nkai = int(np.ceil(self.len/BATCH_SIZE)) # このバッチサイズで繰り返すことができる回数

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
        for i in self.i_rand[self.i_iter:self.i_iter+self.BATCH_SIZE]:
            xi = imread(self.input_file[i])/255. # input読み込み
            yi = imread(self.teach_file[i])/255. # teach読み込み

            # 指定のサイズに変える
            xi = resize(xi,(self.px,self.px),anti_aliasing=True,mode='constant')
            yi = resize(yi,(self.px,self.px),anti_aliasing=True,mode='constant')
            # xi, yi = (1, height, width)
            xi = torch.Tensor(xi[None, :, :])
            yi = torch.Tensor(yi[None, :, :])
            x.append(xi)
            y.append(yi)

        x = torch.stack(x)
        y = torch.stack(y)
        self.i_iter += self.BATCH_SIZE
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

    def gakushuu(self,dalo_train,dalo_test,EPOCH,cnt_multi=10):
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
        for kaime in range(self.mounankai,self.mounankai+EPOCH):
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
                        psnrMSE.append(10*np.log10((1^2)/mse(z,y.to(self.dev)).item()))

                        plt.figure(figsize=[5,4])
                        plt.imshow(z[1].squeeze().to('cpu').detach().numpy(), cmap='gray')
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.save_folder,'denoised_image_DAE_16.png'))
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
            plt.ylabel('MSE')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.loss,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'graph_DAE_16.png'))
            plt.close()

            # PSNRの変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('PSNR(DAE: batch size 16) [dB]')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.psnr,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'psnr_DAE_16.png'))
            plt.close()

    def __call__(self,x,BATCH_SIZE=8):
        self.net.eval()
        x = torch.Tensor(x)
        y = []
        for i in range(0,len(x),BATCH_SIZE):
            y.append(self.net(x[i:i+BATCH_SIZE].to(self.dev)).detach().cpu())
        return torch.cat(y).numpy()


## 実行 ##

train_folder = path.join(path.dirname(__file__), 'train') # 訓練データのフォルダ
test_folder = path.join(path.dirname(__file__), 'test') # 検証データのフォルダ
save_folder = path.join(path.dirname(__file__), 'save') # 結果を保存するフォルダ
cn = 1 # チャネル数 
BATCH_SIZE = 16 # バッチサイズ
px = 128 # 画像の大きさ
EPOCH = 30 # 何回繰り返すか
cnt_multi = 10 # 一回の訓練で何回結果を出力する

# 使うモデルを選ぶ
model = autoencoder


dalo_train = Gazoudalo(train_folder,px,BATCH_SIZE) # 訓練データ
dalo_test = Gazoudalo(test_folder,px,BATCH_SIZE) # 検証データ
dino = Dinonet(model,cn,save_folder)
# 学習開始
dino.gakushuu(dalo_train,dalo_test,EPOCH,cnt_multi)
print("finish")
