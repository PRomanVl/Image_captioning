# Image captioning telegram bot 

### Description
google colab: https://drive.google.com/file/d/1SM1XmjQ83ZdroNAb4RRfTqm_Ko6XeIUW/view?usp=drive_link
Image captioning - this is the task of generating a text description for an image. 
Convolutional CNN neural network models work best with images, and recurrent RNN (or LSTM) models work best with text. 
Therefore, to create a model, it is necessary to combine both of these approaches.

The first model will take an image and produce a vector representation, the second model will take it and generate text.

![model](https://github.com/PRomanVl/Image_captioning/assets/96573887/d2d8c82a-5fe1-4055-9cfc-44538da6183d)




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

### Example:
Input:
![photo_2023-08-07_16-32-52](https://github.com/PRomanVl/Image_captioning/assets/96573887/7acfc772-b69b-4509-bc21-7cf49be2768b)


Output:
a woman is holding a refrigerator door open .

