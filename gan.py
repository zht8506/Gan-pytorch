import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

#---------------维度变化以mnist数据集为例------------------#


os.makedirs("images", exist_ok=True)  #建立“image”文件夹

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)  # (1,28,28)

cuda = True if torch.cuda.is_available() else False

# ----------------生成器----------------------#
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # block：线性层 + BN层（可选）+激活层
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))), # np.prod计算所有元素的乘积，1*28*28=764
            nn.Tanh()
        )

    def forward(self, z):
        # z.shape [batch_size,100]
        img = self.model(z)  # 100->128->256->512->1024->
        img = img.view(img.size(0), *img_shape)  # img.shape: [batch_size,1,28,28]
        return img

# ----------------判别器----------------------#
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # img_flat.shape:[batch_size,784]
        validity = self.model(img_flat) # validity.shape:[batch_size,1]，每一个样本是真实数据的可能性

        return validity


# Loss function
# 损失函数，采用BCE损失，二值交叉熵损失，因为判别器是二分类问题
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
# 初始化生成器与判别器
generator = Generator()
discriminator = Discriminator()


# 将模型放入cuda，以便使用gpu
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# 构建数据集，会从torchvision下载相关数据到本路径目录下的data文件夹
# 如果不能科学上网，可能下载较慢甚至报错，可以将数据集提前下载好放在data文件夹下
# 加载数据集时，对数据集做了transforn，分别是图像尺寸变化Resize、图像变成Tensor、图像数值归一化
os.makedirs("data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
# G和D的优化器
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    # 与Gan论文不同的是，Gan是先训练判别器k次，再训练生成器
    # 而本代码是先训练生成器再训练判别器，并且是各训练一次
    for i, (imgs, _) in enumerate(dataloader):

        # img.shape: [batch_size,1,28,28]
        # Adversarial ground truths
        # valid、fake : shape-[batch_size,1]
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))  # [batch_size,1,28,28]

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  #生成器梯度清零

        # Sample noise as generator input
        # 利用np.random生成随机数据  shape-[batch_size, opt.latent_dim]  默认[64,100]
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)  # [64,1,128,128]

        # Loss measures generator's ability to fool the discriminator
        # 将生成的图像放入判别器进行判别，与1计算其二值交叉熵损失
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        #  训练判别器
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        # 计算判别器对真实样本与生成样本的损失
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i #计算已经迭代了多少个batch-size数
        # 按照采样间隔，将生成的图片保存
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)