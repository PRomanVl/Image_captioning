# Image captioning telegram bot 

### Description

Image captioning - this is the task of generating a text description for an image. 
Convolutional CNN neural network models work best with images, and recurrent RNN (or LSTM) models work best with text. 
Therefore, to create a model, it is necessary to combine both of these approaches.

The first model will take an image and produce a vector representation, the second model will take it and generate text.

![Architecture](https://github.com/PRomanVl/deep_learning_school/blob/main/Image_captionong/img/model.png)




## Telegram bot
@PR_image_captioning_bot  [link](https://t.me/PR_image_captioning_bot)


[Telegram bot code]
### Libraries:
- aiogram==2.17.1
- torch==1.10.0+cpu
- torchvision==0.11.1+cpu


### How it works:
- 1 Telegram bot **bot.py** gets the image 
- 2 Saves to folder 
- 3 **app.py** gets img
  - 3.1 Transform
  - 3.2 Inceptionv3 makes image vector
  - 3.3 RNN model gives description 
- 4 Bot gives to the user predicted description
- 5 Deletes the image


