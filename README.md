# Generate-drum-4bars-w-MusicVAE-n-Groove-midi-dateset
 
## 진행기간 
- 2022.12.26. ~ 2022.12.28

## 목적
- **논문을 기반으로 MusicVAE, Groove midi dataset을 사용하여 4마디의 드럼 샘플 생성**  
          
## 코드 설명

   
코드     | 코드 링크   | 
:-------:|:-----------:|
Seq2Seq(LSTM)|[Seq2Seq(LSTM)](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq(LSTM).ipynb)|         
Seq2Seq with Attention | [Seq2Seq with Attention](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq%20with%20Attention.ipynb)|
Seq2Seq with Attention(Bi-LSTM)| [Seq2Seq with Attention(Bi-LSTM)](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Seq2Seq%20with%20Attention(Bi-LSTM).ipynb)| 
Transformer| [Transformer](https://github.com/AliceecilA93/Machine-Translation-with-LSTM-and-Transformer/blob/main/Transformer.ipynb) |
        

## 사용된 데이터  

- Groove MIDI dataset [groove-v1.0.0-midionly.zip](https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip)
- Groove MIDI -over 2bar dataset(custom) [groove-v1.0.0-midionly_over_2bar.zip](https://drive.google.com/file/d/1JV2IryZOZJmjSisdxGk6iaq0Di6YulDm/view?usp=share_link)


## 사용된 모델 

- VAE(Variational AutoEncoder) 

![image](https://user-images.githubusercontent.com/112064534/209743089-dee3bf19-271b-47ab-8333-fbedac55ab2f.png)



INPUT x값의 특징을 추출하여 잠재백터(latent vector)인 z에 담은 후 z를 통해 x의 특징이 드러난 새로운 데이터 생성 



  - Bi-directional LSTM Encoder
  - Hierarchical LSTM decoder
  - Groove LSTM decoder


## 과정  

 1. 개발환경 : Python, Tensorflow, Colab
 
 2. 데이터 전처리
    - magenta/scripts/convert_dir_to_note_sequences.py 에서 convert_directory 함수 사용
      ==> Directory 전체를 변환
    - info.csv를 통해 ['duraton']이 4초이상만 따로 뽑아서 데이터셋 생성
      ==> 논문에서 제시한 Hierarchical decoder가 sequences with long-term structure의 문제를 
          해결하는지 보기 위함


 3. 데이터셋
   
 데이터셋 | 데이터 갯수 | 
 :-------:|:-----------:|
 Groove MIDI Dataset | 1,150 |        
 Groove MIDI- over 2bar Dataset | 557 |

 
 4. 모델 Config 
 
 - GrooVAE : 드럼 샘플을 생성하고 조절하기위한 MusicVAE의 변종  
     
   * ['groovae_4bar'] 
 
         magenta/models/music_vae/config.py에서 ['groovae_4bar'] config default값 그대로 사용
    
  
    
   * ['groovae_4bar'] with changing hparams
    
         magenta/models/music_vae/config.py에서 ['groovae_4bar'] config를 논문과 비슷하게 적용 
    
         1) Change size of Encoder 
         => 논문에서 사용된 데이터셋은 16마디로 인코더 순환층은 [2048,2048]로 사용. 본 과제의 목적은 드럼 4마디를 생성해내는 것임으로 1/4을 적용하여 인코더 순환층은 [512,512]로 변경
    
         2) learning rate = 0.001
    
         3) min_learning_rate = 0.00001
    
         4) decay_rata = 0.9999
    
   * ['hierdec-mel_4bar']
    
         magenta/models/music_vae/config.py에서 ['hierdec-mel_16bar'] config를 차용하여 데이터셋에 맞게 변경 
    
         1) Change level_lengths 
    
         논문에서 사용한 것은 1 마디당 16개의 음표를 사용하고 최대 16마디를 사용하여 level_lengths는 [16,16] 이였지만 본 과제는 4마디를 뽑는 것으로 4마디로 사이즈 조정 [16,4]
   
         2) Chance max_seq_len
    
         위와 같은 이유로 256에서 16*4(64)로 변경 
    
         3) learning rate = 0.001
    
         4) min_learning_rate = 0.00001
    
         5) decay_rata = 0.9999
         
         6) Change data_converter
         
         논문에서는 mel_16bar_converter 사용했지만 현재 데이터셋은 드럼으로만 구성되어있어 
         GrooveConverter로 변경 
         
   

## 결과
- ['groovae_4bar'] 보다 ['groovae_4bar'] with changing hprams가 좀 더 자연스러움. 
  (batch_size = 1) 
 * ['groovae_4bar'] [drum_4bar]()
 * ['groovae_4bar'] with changing hparams [drum_4bar_1] ()

- 


형태소 분석기는 5가지 중에서 Mecab을 이용하고, Embedding Dimension은 높으면 높을수록, LSTM Layers는 논문에 나온 4층보다는 3층이, Dropout은 0.2 , Recurrent Dropout은 사용하지 않는 것이, Hidden Units은 낮을수록 성능이 더 좋았다. 

하지만, 결과적으로 Transformer가 BiLSTM보다 Accuracy가 낮았음에도 더 좋은 성능을 보여주었는데 
첫 번째, LSTM은 Input을 순차적으로 받아 병렬처리가 어려운 반면, Transformer는 Sequence를 한번에 넣음으로써 병렬처리가 가능해졌기 때문이고
두 번째, LSTM의 Attention은 Encoder의 Hidden state와 Decoder의 Hidden state를 연산하여 Attention score를 계산하는 반면, Transformer의 Self-Attention은 Encoder로 들어간 벡터와 Encoder로 들어간 모든 벡터를 연산해서 Attention score를 계산함으로써 각 input data들의 유사도를 추정할 수 있기 때문에 
성능이 더 높은 이유이다. 

    


## 참조
-Roberts, Adam, et al. "A hierarchical latent vector model for learning long-term structure in music." International conference on machine learning. PMLR, 2018.
https://arxiv.org/pdf/1803.05428.pdf
-Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck, and David Bamman.
"Learning to Groove with Inverse Sequence Transformations."
  International Conference on Machine Learning (ICML), 2019.
- https://github.com/magenta/magenta/tree/main/magenta/models/music_vae
- https://github.com/oobinkim/MusicVAE_Groove
