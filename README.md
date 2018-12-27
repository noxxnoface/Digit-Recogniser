# Digit Recognition
 Recognizing digits using keras with tensorflow backend producing test accuracy of 98% on online platform Kaggle<br>

 The varying accuracies corresponding to epochs and batch size can be observed below.<br>

 #---------------------<br>
 model.fit(X_train, y_train, epochs = 1, batch_size = 1,verbose = 1)<br>
 
 Epoch 1/1<br>
 
 42000/42000 [==============================] - 1230s 29ms/step - loss: 0.1537 - acc: 0.9542<br>
 
 #---------------------<br>
 
 model.fit(X_train, y_train, epochs = 3, batch_size = 1000,verbose = 1)<br>
 Epoch 1/3<br>
 42000/42000 [==============================] - 260s 6ms/step - loss: 0.0694 - acc: 0.9795<br>
 Epoch 2/3<br>
 42000/42000 [==============================] - 290s 7ms/step - loss: 0.0585 - acc: 0.9831<br>
 Epoch 3/3<br>
 42000/42000 [==============================] - 291s 7ms/step - loss: 0.0533 - acc: 0.9843<br>
 
 #----------------------<br>
 
 model.fit(X_train, y_train, epochs = 10, batch_size = 32,verbose = 1)
 Epoch 1/10<br>
 42000/42000 [==============================] - 279s 7ms/step - loss: 0.0490 - acc: 0.9856<br>
 Epoch 2/10<br>
 42000/42000 [==============================] - 193s 5ms/step - loss: 0.0345 - acc: 0.9898<br>
 Epoch 3/10<br>
 42000/42000 [==============================] - 210s 5ms/step - loss: 0.0282 - acc: 0.9913<br>
 Epoch 4/10<br>
 42000/42000 [==============================] - 284s 7ms/step - loss: 0.0231 - acc: 0.9929<br>
 Epoch 5/10<br>
 42000/42000 [==============================] - 267s 6ms/step - loss: 0.0201 - acc: 0.9938<br>
 Epoch 6/10<br>
 42000/42000 [==============================] - 252s 6ms/step - loss: 0.0164 - acc: 0.9945<br>
 Epoch 7/10<br>
 42000/42000 [==============================] - 183s 4ms/step - loss: 0.0135 - acc: 0.9957<br>
 Epoch 8/10<br>
 42000/42000 [==============================] - 276s 7ms/step - loss: 0.0106 - acc: 0.9966<br>
 Epoch 9/10<br>
 42000/42000 [==============================] - 181s 4ms/step - loss: 0.0091 - acc: 0.9970<br>
 Epoch 10/10<br>
 42000/42000 [==============================] - 171s 4ms/step - loss: 0.0091 - acc: 0.9969<br>
 
 #----------------------<br>
 
 model.fit(X_train, y_train, epochs = 20, batch_size = 64,verbose = 1)<br>
 Epoch 1/20<br>
 42000/42000 [==============================] - 42s 1ms/step - loss: 0.2832 - acc: 0.9164<br>
 Epoch 2/20<br>
 42000/42000 [==============================] - 49s 1ms/step - loss: 0.0802 - acc: 0.9749<br>
 Epoch 3/20<br>
 42000/42000 [==============================] - 45s 1ms/step - loss: 0.0596 - acc: 0.9807<br>
 Epoch 4/20<br>
 42000/42000 [==============================] - 49s 1ms/step - loss: 0.0472 - acc: 0.9851<br>
 Epoch 5/20<br>
 42000/42000 [==============================] - 38s 909us/step - loss: 0.0400 - acc: 0.9870<br>
 Epoch 6/20<br>
 42000/42000 [==============================] - 33s 791us/step - loss: 0.0305 - acc: 0.9900<br>
 Epoch 7/20<br>
 42000/42000 [==============================] - 33s 789us/step - loss: 0.0280 - acc: 0.9910<br>
 Epoch 8/20<br>
 42000/42000 [==============================] - 33s 790us/step - loss: 0.0239 - acc: 0.9924<br>
 Epoch 9/20<br>
 42000/42000 [==============================] - 40s 947us/step - loss: 0.0206 - acc: 0.9935<br>
 Epoch 10/20<br>
 42000/42000 [==============================] - 44s 1ms/step - loss: 0.0185 - acc: 0.9936<br>
 Epoch 11/20<br>
 42000/42000 [==============================] - 47s 1ms/step - loss: 0.0154 - acc: 0.9951<br>
 Epoch 12/20<br>
 42000/42000 [==============================] - 49s 1ms/step - loss: 0.0148 - acc: 0.9954<br>
 Epoch 13/20<br>
 42000/42000 [==============================] - 46s 1ms/step - loss: 0.0142 - acc: 0.9955<br>
 Epoch 14/20<br>
 42000/42000 [==============================] - 47s 1ms/step - loss: 0.0085 - acc: 0.9972<br>
 Epoch 15/20<br>
 42000/42000 [==============================] - 53s 1ms/step - loss: 0.0099 - acc: 0.9966<br>
 Epoch 16/20<br>
 42000/42000 [==============================] - 39s 925us/step - loss: 0.0101 - acc: 0.9969<br>
 Epoch 17/20<br>
 42000/42000 [==============================] - 33s 787us/step - loss: 0.0088 - acc: 0.9971<br>
 Epoch 18/20<br>
 42000/42000 [==============================] - 36s 851us/step - loss: 0.0083 - acc: 0.9972<br>
 Epoch 19/20<br>
 42000/42000 [==============================] - 33s 782us/step - loss: 0.0061 - acc: 0.9980<br>
 Epoch 20/20<br>
 42000/42000 [==============================] - 33s 788us/step - loss: 0.0083 - acc: 0.9970<br>
 
