# Amazon reviews classification

[See notebook at Google Colab](https://colab.research.google.com/drive/1aMjB7TjgsOWnQmXCUNyjqRhWopKmAEfH?usp=sharing)

The aim of this notebook is to practise basic text processing using scikit-learn and Keras.

I was using previously extracted with my [amazon-reviews-scraper](https://github.com/pai-pai/amazon-reviews-scraper) reviews of 21 products. The combined and cleaned dataset contains 56142 entries.

<img src="https://pai-pai-github-images.s3.amazonaws.com/amazon-reviews-classification-data-sample.png" alt="cleaned-data-sample" />

---

I applied scikit-learn's Logistic Regression classifier with two types of vectorizers: Count Vectorizer and TF-IDF Vectorizer (with different parameters). The obtained AUC values are shown below:
- 0.831 -- CountVectorizer.
- 0.836 -- TfidfVectorizer.
- 0.840 -- TfidfVectorizer with n-grams.
- 0.838 -- TfidfVectorizer with n-grams and top K-features.

---

My next step was building model with Keras.
I started from basic Sequential model with 4 layers:
```
Model: "sequential_15"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 50)                48190600  
                                                                 
 dense_45 (Dense)            (None, 16)                816       
                                                                 
 dense_46 (Dense)            (None, 16)                272       
                                                                 
 dense_47 (Dense)            (None, 1)                 17        
                                                                 
=================================================================
Total params: 48,191,705
Trainable params: 48,191,705
Non-trainable params: 0
_________________________________________________________________
```

After this I added Dropout levels to deal with over-fitting:
```
Model: "sequential_16"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 keras_layer (KerasLayer)    (None, 50)                48190600  
                                                                 
 dense_48 (Dense)            (None, 16)                816       
                                                                 
 dropout (Dropout)           (None, 16)                0         
                                                                 
 dense_49 (Dense)            (None, 16)                272       
                                                                 
 dropout_1 (Dropout)         (None, 16)                0         
                                                                 
 dense_50 (Dense)            (None, 1)                 17        
                                                                 
=================================================================
Total params: 48,191,705
Trainable params: 48,191,705
Non-trainable params: 0
_________________________________________________________________
```

Following that I combined using of Dropout layer and adding L2 regularization to Dense layers.

There are results:
- loss: 0.394, accuracy: 0.8790 -- Base model.
- loss: 0.4119, accuracy: 0.8816 -- With added Dropout layers.
- loss: 0.430, accuracy: 0.881 -- With L2 regularization only.
- loss: 0.404 - accuracy: 0.883 -- Final model with Dropout and L2 regularization.

---

The last part of the notebook is a building RNN and SCNN.

Results of Recurrent Neural Network:
```
Model: "sequential_32"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 500, 200)          4000000   
                                                                 
 spatial_dropout1d (SpatialD  (None, 500, 200)         0         
 ropout1D)                                                       
                                                                 
 lstm (LSTM)                 (None, 100)               120400    
                                                                 
 dense_93 (Dense)            (None, 1)                 101       
                                                                 
=================================================================
Total params: 4,120,501
Trainable params: 4,120,501
Non-trainable params: 0
_________________________________________________________________
```
loss: 0.271, accuracy: 0.902

Separable Convolutional Neural Network:
```
Model: "sequential_34"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_2 (Embedding)     (None, 500, 200)          4000000   
                                                                 
 dropout_26 (Dropout)        (None, 500, 200)          0         
                                                                 
 separable_conv1d (Separable  (None, 500, 64)          13464     
 Conv1D)                                                         
                                                                 
 separable_conv1d_1 (Separab  (None, 500, 64)          4352      
 leConv1D)                                                       
                                                                 
 max_pooling1d (MaxPooling1D  (None, 166, 64)          0         
 )                                                               
                                                                 
 separable_conv1d_2 (Separab  (None, 166, 128)         8512      
 leConv1D)                                                       
                                                                 
 separable_conv1d_3 (Separab  (None, 166, 128)         16896     
 leConv1D)                                                       
                                                                 
 global_average_pooling1d (G  (None, 128)              0         
 lobalAveragePooling1D)                                          
                                                                 
 dropout_27 (Dropout)        (None, 128)               0         
                                                                 
 dense_95 (Dense)            (None, 1)                 129       
                                                                 
=================================================================
Total params: 4,043,353
Trainable params: 4,043,353
Non-trainable params: 0
_________________________________________________________________
```
loss: 0.3407, accuracy: 0.8931

____

## Technical stack
- pandas
- scikit-learn
- TensorFlow
