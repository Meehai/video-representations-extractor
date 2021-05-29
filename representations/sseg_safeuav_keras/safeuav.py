from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, Activation, add, concatenate, Conv2DTranspose
from typing import Tuple

def get_unet_MDCB_with_deconv_layers(input_shape:Tuple[int, int, int], init_nb:int, num_classes:int):
	inputs = Input(input_shape)
	
	down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(inputs)
	down1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(down1)
	down1pool = Conv2D(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(down1)
	
	down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down1pool)
	down2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(down2)
	down2pool = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(down2)

	down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down2pool)
	down3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(down3)
	down3pool = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(down3)

	# stacked dilated convolution
	dilate1 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=1)(down3pool)
	dilate2 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=2)(dilate1)
	dilate3 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=4)(dilate2)
	dilate4 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=8)(dilate3)
	dilate5 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=16)(dilate4)
	dilate6 = Conv2D(init_nb*8, (3, 3), activation='relu', padding='same', dilation_rate=32)(dilate5)
	dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])

	up3 = Conv2DTranspose(init_nb*4, (3, 3), activation='relu', padding='same', strides=(2, 2))(dilate_all_added)
	up3 = concatenate([down3, up3])
	up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)
	up3 = Conv2D(init_nb*4, (3, 3), activation='relu', padding='same')(up3)

	up2 = Conv2DTranspose(init_nb*2, (3, 3), activation='relu', padding='same', strides=(2, 2))(up3)
	up2 = concatenate([down2, up2])
	up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)
	up2 = Conv2D(init_nb*2, (3, 3), activation='relu', padding='same')(up2)

	up1 = Conv2DTranspose(init_nb, (3, 3), activation='relu', padding='same', strides=(2, 2))(up2)
	up1 = concatenate([down1, up1])
	up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)
	up1 = Conv2D(init_nb, (3, 3), activation='relu', padding='same')(up1)

	classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

	model = Model(inputs=inputs, outputs=classify, name='MSMT-Stage-1-TransposeConvs')
	return model
