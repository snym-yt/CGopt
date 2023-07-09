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

NUM_SAMPLES = 1000
ENV_NAME = "ALE/Breakout-v5"
MINI_BATCH_SIZE = 8
LEARNING_RATE_GENERATOR = 4e-5
LEARNING_RATE_DISCRIMINATOR = 4e-5
EPOCHS = 30
CLIP_WEIGHTS = 0.01
GENER_FILTERS = 1

# Hyperparameter for saving the models in the training process
# BASE_PATH = "/" 
# CREATE_SAVES = True
# LOAD_SAVES = False


## ニューラルネットワークモデル ##

class GAN(nn.Module):
    def __init__(self, input_shape, output_shape, cn):
        super(GAN, self).__init__()
        self.cn = cn
        self.generator = Generator(input_shape, output_shape)
        self.discriminator = Discriminator(input_shape)

    def forward(self, x):
        generated_images = self.generator(x)
        reconstructed_images = self.discriminator(generated_images)
        return reconstructed_images

class Generator(nn.Module):
  def __init__(self, input_shape=[8, 1, 128, 128], output_shape=[8, 1, 128, 128]):
    super().__init__()
    self.conv = nn.Sequential(
        nn.ConvTranspose2d(input_shape[0], GENER_FILTERS * 8, kernel_size=4, stride=1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(GENER_FILTERS * 8),

        nn.ConvTranspose2d(GENER_FILTERS * 8, GENER_FILTERS * 4, kernel_size = 4, stride = 2, padding = 1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(GENER_FILTERS * 4),

        nn.ConvTranspose2d(GENER_FILTERS * 4, GENER_FILTERS * 2, kernel_size = 4, stride = 2, padding = 1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(GENER_FILTERS * 2),

        nn.ConvTranspose2d(GENER_FILTERS * 2, GENER_FILTERS, kernel_size = 4, stride = 2, padding = 1),
        nn.LeakyReLU(),
        nn.BatchNorm2d(GENER_FILTERS)
    )
    print("self.conv(Generator):OK\n")


    conv_out_size = self._get_conv_out(input_shape)
    print("conv_out_size(Generator):OK\n")
    print("conv_out_size(Generator):", conv_out_size)

    self.out = nn.Sequential(
        nn.Linear(conv_out_size, 1012),
        nn.LeakyReLU(),
        nn.Linear(1012, np.prod(output_shape[1:])),  # プロダクトの引数から最初の次元を削除
        nn.Tanh()
    )

    self.output_shape = [output_shape[0], *output_shape[2:]]  # output_shapeの形状を修正
    """
    self.out = nn.Sequential(
        nn.Linear(conv_out_size, 1012),
        nn.LeakyReLU(),
        nn.Linear(1012, np.prod(output_shape[1:])),  # プロダクトの引数から最初の次元を削除
        nn.Tanh()
    )

    self.output_shape = [output_shape[0], *output_shape[2:]]  # output_shapeの形状を修正
    """

  def _get_conv_out(self, shape):
    out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(out.size()))

  def forward(self, input):
    conv_out = self.conv(input).view(input.shape[0], -1)
    return self.out(conv_out.view((-1, conv_out.size(1), *self.output_shape[2:])))


  # def save(self):
  #   if not os.path.exists(BASE_PATH):
  #     os.makedirs(BASE_PATH)
  #   # torch.save(self.state_dict(), BASE_PATH + "/generator.pt")

  # def load(self):
  #   self.load_state_dict(torch.load(BASE_PATH + "/generator.pt"), strict=False)


class Discriminator(nn.Module):
  def __init__(self, input_shape):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size = 8, stride = 4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3,stride = 1)
    )

    conv_out_size = self._get_conv_out(input_shape)

    self.out = nn.Sequential(
        nn.Linear(conv_out_size, 1012),
        nn.ReLU(),
        nn.Linear(1012, 1),
        # nn.Dropout(0.005),
        nn.Sigmoid()
    )

  def _get_conv_out(self, shape):
    out = self.conv(torch.zeros(1, *shape))
    return int(np.prod(out.size()))

  def forward(self, input):
    conv_out = self.conv(input).view(input.shape[0], -1)
    return self.out(conv_out).squeeze(-1)
"""""
  def save(self):
    if not os.path.exists(BASE_PATH):
      os.makedirs(BASE_PATH)
    torch.save(self.state_dict(), BASE_PATH + "/discriminator.pt")

  def load(self):
    self.load_state_dict(torch.load(BASE_PATH + "/discriminator.pt"), strict=False)
"""""
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

        self.input_shape = [1, 128, 128]  # 入力データの形状を設定する
        self.output_shape = [1, 128, 128]  # 出力データの形状を設定する
        self.net = net(self.input_shape, self.output_shape, self.cn)

        # self.net = net()
        # self.opt = torch.optim.Adam(self.net.parameters(),lr=learning_rate)
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
                lossMSE = mse(z,y.to(self.dev)) # 訓練データの損失
                # self.opt.zero_grad()
                lossMSE.backward()
                # self.opt.step()

                # 検証データにテスト
                if((i_batch+1)%int(np.ceil(dalo_train.nkai/cnt_multi))==0 or i_batch==dalo_train.nkai-1):
                    self.net.eval()
                    lossMSE = []
                    psnrMSE = []
                    if(n_kaku):
                        gazou = []
                        n_kaita = 0

                    for x,y in test_data:
                        z = self.net(x.to(self.dev))
                        # 検証データの損失
                        lossMSE.append(mse(z,y.to(self.dev)).item())
                        psnrMSE.append(10*np.log10((1^2)/mse(z,y.to(self.dev)).item()))
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
                    lossMSE = np.mean(lossMSE)
                    psnrMSE = np.mean(psnrMSE)
                    # if(n_kaku):
                    #     gazou = np.hstack(gazou)
                    #     # imsave(os.path.join(self.save_folder,'kekka%03d.jpg'%(kaime+1)),gazou)

                    # 今の状態を出力する
                    print('%d:%d/%d ~ 損失:%.4e %.2f分過ぎた'%(kaime+1,i_batch+1,dalo_train.nkai,lossMSE,(time.time()-t0)/60))
                    print('PSNR = {}'.format(psnrMSE))
                    self.net.train()

            # ミニバッチ一回終了
            self.loss.append(lossMSE)
            self.psnr.append(psnrMSE)
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
            plt.savefig(os.path.join(self.save_folder,'graph_GAN.png'))
            plt.close()

            # PSNRの変化を表すグラフを書く
            plt.figure(figsize=[5,4])
            plt.xlabel('trial')
            plt.ylabel('PSNR(GAN)')
            ar = np.arange(1,kaime+2)
            plt.plot(ar,self.psnr,'#11aa99')
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_folder,'psnr_GAN.png'))
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
model = GAN


dalo_train = Gazoudalo(train_folder,px,n_batch) # 訓練データ
dalo_test = Gazoudalo(test_folder,px,n_batch) # 検証データ
dino = Dinonet(model,cn,save_folder)
# 学習開始
dino.gakushuu(dalo_train,dalo_test,n_loop,n_kaku,cnt_multi)
print("finish")
