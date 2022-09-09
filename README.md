# OnlyFontGenerator
- 한 글자로 글씨체를 분석하는 인공지능.

Model Structure
-----------------------------
- U-Net based model structure      
- Conv Block that in the Content Extractor can be translated as Conv (in_channels) (out_channels) k(kernel_size)s(stride)p(padding).
    
![image](https://user-images.githubusercontent.com/66504341/185335120-b5d95f0f-f0b2-425f-9eb4-1a12bfecf626.png)

Loss Function
------------------------------
![image](https://user-images.githubusercontent.com/66504341/185335174-c1917045-8c6c-4372-a61e-3d8217ea0615.png)


Train
------------------------------
- hyper parameters, when epoch 0 ~ 15    
    
![image](https://user-images.githubusercontent.com/66504341/185335217-c1ce699e-3ec0-4d62-bba2-ae46cdadcf03.png)    
    
    
    
    
- hyper parameters, when epoch 16 ~ 80    
    
![image](https://user-images.githubusercontent.com/66504341/185335233-b6c5f24f-ec41-4d05-b934-95ed1717187f.png)    
    
    
    
    
- history    
    
![image](https://user-images.githubusercontent.com/66504341/185335248-f220b6f2-2465-4193-a6cf-4dad172ca64d.png)    
    
    
    
    
- predicted result (first: input letter, second: gothic letter, third: label letter, fourth: predicted result)    
    
![pred](https://user-images.githubusercontent.com/66504341/185337804-bda2d5c1-ccb2-44f0-bac7-301627bc4557.gif)


Test
-----------------------------
- interpolation    
    
![interpolate](https://user-images.githubusercontent.com/66504341/185338801-35bc5aa1-051a-4117-8a46-5e29f8495ccd.gif)    
    
    
    
    
- my handwriting (left: my handwriting, right: predicted result)    
    
![image](https://user-images.githubusercontent.com/66504341/185335284-4e92b94a-3186-4e3d-8572-a02dbbcad06f.png)


Dataset
-----------------------
- [handwriting dataset](https://clova.ai/handwriting/list.html)    
- [gothic dataset](https://hangeul.naver.com/2021/fonts/nanum)    


Acknowledgements
--------------------------
- [GAN-handwriting-styler](https://github.com/jeina7/GAN-handwriting-styler) by [jeina7](https://github.com/jeina7)    
- [zi2zi](https://github.com/kaonashi-tyc/zi2zi) by [kaonash-tyc](https://github.com/kaonashi-tyc)
