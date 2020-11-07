
<img src="icons/dlknock_icon.png" width=300>

# ディープラーニング∞CheatSheet!!

DeepLearningの実装Tips (WiP)

**ノック作るのが大変になってきたのでTips集に変更しました**

- 【注意】このページを利用して、または関して生じた事に関しては、**私は一切責任を負いません。** すべて **自己責任** でお願い致します。
- なんとなく本とか買わずにDLを勉強したいーーーって人向けだと思う

もしこれがみなさんのお役に立ったらGithub Sponsorになってください！

~~何問になるか分からないので∞本になってます。多分これからいろんな技術が出るからどんどん更新する予定でっす。これはイモリと一緒にディープラーニングの基礎からDLのライブラリの扱い、どういうDLの論文があったかを実装しながら学んでいくための問題集です。本とか論文読んだだけじゃ机上の空想でしかないので、ネットワークの作成や学習率などのハイパーパラメータの設定を自分の手を動かしながら勉強するための問題集です。~~

~~**問題集として使ってもテンプレやチートシートとして使っても使い方は自由です！！！！**~~

## Related

- ***Study-AI株式会社様 http://kentei.ai/ のAI実装検定のシラバスに使用していただくことになりました！(画像処理100本ノックも）Study-AI株式会社様ではAIスキルを学ぶためのコンテンツを作成されており、AIを学ぶ上でとても参考になります！
検定も実施されてるので、興味ある方はぜひ受けることをお勧めします！***

- ***画像処理100本ノック!! https://github.com/yoyoyo-yo/Gasyori100knock)***


## Install package

```bash
# pytorch
$ pip install matplotlib opencv-python easydict torch torchvision torchsummary
# tensorflow-2.1
$ pip install matplotlib opencv-python easydict tensorflow==2.1
```


## Tips

### Model

| API | Code |
|:---:|:---:|
| torchvision.models | [pytorch STL10](scripts/api_pytorch.ipynb)
| torchvision.models(VGG16) | [pytorch CIFAR100](pytorch/VGG16_ft_CIFAR100_pytorch.ipynb)

| Method | Code |
|:---:|:---:|
| VGG16| [pytorch](pytorch/VGG16_CIFAR100_pytorch.ipynb)
| VGG19 | [pytorch](pytorch/VGG19_CIFAR100_pytorch.ipynb)
| GoogLeNet-v1 | [ pytorch](Spytorch/googletnetv1_pytorch.ipynb) 
| ResNet-50, 101, 152, 18, 34 | [ pytorch](pytorch/ResNet_CIFAR100_pytorch.ipynb) 
| ResNeXt-50,101 | [ pytorch](pytorch/ResNeXt_CIFAR100_pytorch.ipynb)
| Xception| [ pytorch](pytorch/Xception_CIFAR100_pytorch.ipynb) 
| DenseNet121, 169, 201, 264| [ pytorch](pytorch/DenseNet_CIFAR100_pytorch.ipynb) 
| MobileNet-v1 | [ pytorch](pytorch/MobileNetv1_pytorch.ipynb)
| MobileNet-v2 | [ pytorch](pytorch/MobileNetv2_CIFAR100_pytorch.ipynb)
| EfficientNet | [ pytorch](pytorch/EfficientNet_CIFAR100_pytorch.ipynb)

###  Interpretation

| Method |  Code |
|:---:|:---:|
| Grad-CAM | [ pytorch](pytorch/GradCAM_STL10_pytorch.ipynb)


### Segmentation

| Method | Code |
|:---:|:---:|
| UNet|  [pytorch](pytorch/UNet_Seg_VOC2012_pytorch.ipynb) 

### Object Detection

| Method | Code |
|:---:|:---:|
| MaskRCNN (torchvision)| [pytorch](pytorch/MaskRCNN_torchvision_sample.ipynb) 


### AE
| Method | Code |
|:---:|:---:|
| AE MNIST |  [ pytorch](pytorch/AE_MNIST_pytorch.ipynb)
| AE cifar10 |  [ pytorch](pytorch/AE_CIFAR10_pytorch.ipynb)
| AE |  [ pytorch](pytorch/AE_pytorch.ipynb) 
| AEによる異常検知 | [ (MNIST)pytorch](pytorch/AE_AnormalyDet_MNIST_pytorch.ipynb), [(FashionMNIST) pytorch](pytorch/AE_AnormalyDetection_fashionmnist_pytorch.ipynb)
| ConvAE cifar10 |  [ pytorch](pytorch/ConvAE_CIFAR10_pytorch.ipynb) 
| ConvAE |  [ pytorch](pytorch/ConvAE_pytorch.ipynb)
| VAE MNIST |  [ pytorch](pytorch/VAE_MNIST_pytorch.ipynb)
| VAE + Clustering MNIST | [ pytorch](pytorch/VAE_Clustering_MNIST_pytorch.ipynb) 

### GAN
| Method | Code | 
|:---:|:---:|
| GAN cifar10 | [ pytorch](pytorch/gan_cifar10_pytorch.py) 
| GAN | [ pytorch](pytorch/gan_pytorch.py) 
| DCGAN cifar10 | [ pytorch](pytorch/DCGAN_CIFAR10_pytorch.ipynb) 
| DCGAN | [ pytorch](pytorch/DCGAN_pytorch.ipynb)
| CGAN MNIST | [ pytorch](pytorch/CGAN_MNIST_pytorch.ipynb)
| CGAN CIFAR10 | [ pytorch](pytorch/CGAN_CIFAR10_pytorch.ipynb)
| pix2pix Seg | [ pytorch](pytorch/Pix2pix_Seg_pytorch.ipynb) [ tf.keras](scripts_tf_keras/pix2pix_tf2.1_keras.py)
| WGAN CIFAR10 | [ pytorch](pytorch/WGAN_CIFAR10_pytorch.ipynb)
| WGAN | [ pytorch](pytorch/WGAN_pytorch.ipynb)
| WGAN-GP CIFAR0 | [ pytorch](pytorch/WGANGP_CIFAR10_pytorch.ipynb) 
| WGAN-GP | [ pytorch](pytorch/WGANGP_pytorch.ipynb) 
| alphaGAN MNIST | [ pytorch](pytorch/alphaGAN_mnist_pytorch.py)
| alphaGAN cifar10 | [ pytorch](pytorch/alphaGAN_cifar10_pytorch.py)
| CycleGAN | [ pytorch](pytorch/CycleGAN_pytorch.ipynb) 

### Other
| Method | Code |
|:---:|:---:|
| Style Transfer|  [tf.keras](tf/StyleTransfer_tf2.1_keras.py) |


### NLP
| Method | Code | 
|:---:|:---:|
| seq2seq | [ pytorch](pytorch/Seq2seq_pytorch.ipynb)
| Transformer | [ pytorch](pytorch/Transformer_pytorch.ipynb)
| HRED | [ pytorch](pytorch/HRED_pytorch_sand.ipynb) 
| Word2Vec (Skip-gram)| [ pytorch](pytorch/Word2vec_pytorch.ipynb) |

## Update

Twitterで更新を発信してますぅ

https://twitter.com/curry_frog

- 2020.5.3 Sun [pytorch] CycleGANを追加
- 2020.4.3 Fri [tf.keras] pix2pixを追加
- 2020.3.27 Thu [tf.keras] Style Transferを追加
- 2020.2.25 Tue [Pytorch] WGAN-GPを修正
- 2020.1.1 [Pytorch] EfficientNetB1~B7を追加
- 2019.12.30 [Pytorch] EfficientNetB0を追加
- 2019.12.23 Chainerのサポートが終了したらしいので、PytorchとTensorflowに絞っていきます
- 2019.12.23 [Pytorch] 可視化 Grad-CAMを追加
- 2019.11.23 [Pytorch] 言語処理・会話生成のHREDを追加
- 2019.11.19 [Pytorch] 画像生成のWGAN-GPを追加
- 2019.11.8 [Pytorch]　画像生成のVAEとalphaGANを追加
- 2019.10.28 [Pytorch] 画像生成のWGANを追加
- 2019.10.21 [PyTorch] Semantic SegmentationでSegNetを追加
- 2019.10.16 [PyTorch] Seq2Seq Hard Attentionを追加
- 2019.10.10 [PyTorch] Seq2Seq Attention(Step別)を追加
- 2019.9.30 [Pytorch] MobileNet v2 を追加
- 2019.9.19 [TensorFlow] Xception, MobileNet_v1 を追加
- 2019.9.16 [TensorFlow] ResNet 18, 34, 50, 101, 152 を追加
- 2019.8.19 [Pytorch] NLP: Seq2seq+Attention, word2vecを追加
- 2019.8.15 [Pytorch] pix2pixを追加
- 2019.8.4 [Pytorch] DenseNet121, 169, 201, 264を追加
- 2019.7.30 [PyTorch, Keras] Xceptionを追加
- 2019.7.28 [Keras] ResNeXt-50, 101を追加
- 2019.7.23 [Pytorch] ResNeXt-50, 101を追加
- 2019.7.17 [Pytorch] VAEを追加  [keras, tensorflow, chainer] CGAN(MNIST)を追加
- 2019.7.5 [pytorch, keras]ResNet18, 34, 101, 152を追加
- 2019.6.16 [pytorch, tensorflow, keras, chainer] ResNet50を追加
- 2019.6.9 [tensorflow] DCGANを追加
- 2019.6.7 [Pytorch, tensorflow, keras, chainer]GoogleNet-v1(Inception)を追加
- 2019.5.26 [tensorflow] DCGAN, Conditional GANを追加
- 2019.5.19 [Keras, Chainer] ConditionalGANを追加
- 2019.5.18 [データセット準備] MNIST, [Pytorch]ConditionalGANを追加
- 2019.5.2 [データセット準備] Cifar10、[AutoEncoder, ConvAutoEncoder, GAN, DCGAN]Cifar10を追加
- 2019.3.31 [画像認識モデル] APIを追加
- 2019.3.19 [Pytorch][Chainer] GAN, DCGANを追加
- 2019.3.17 Pooling layerを追加したけど、あとからクラス化と学習を追加する予定
- 2019.3.17 seq2seq, convolutional layer を追加
- 2019.3.16 ニューラルネットをクラス化　を追加
- 2019.3.13 パーセプトロン系を追加
- 2019.3.12 AutoEncoder, ConvAutoEncoder, パーセプトロンを追加
- 2019.3.9 GAN, DCGANを追加
- 2019.3.6 RNN, LSTM, BDLSTMを追加
- 2019.3.5 AutoEncoder, RNNを追加　
- 2019.3.4 データ拡張・回転を追加
- 2019.3.3 UNetを追加

## Citation

```bash
@article{yoyoyo-yoDeepLearningMugenKnock,
    Author = {yoyoyo-yo},
    Title = {DeepLearningMugenKnock},
    Journal = {https://github.com/yoyoyo-yo/DeepLearningMugenKnock},
    Year = {2019}
}
```

## License

&copy; Curry yoshi All Rights Reserved.

This is under MIT License.
