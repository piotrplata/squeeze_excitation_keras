#channel squeeze, spatial excitation
def sSE(input_features):
    squeezed = Conv2D(1,kernel_size=(1,1),strides=(1,1),padding='same',activation="sigmoid", use_bias="false")(input_features)
    excited = multiply([squeezed,input_features])
    return excited

#spatial squeeze, channel excitation
def cSE(input_features, ratio = 2):
    
    filters = int(input_features.get_shape()[-1])
    squeezed = GlobalAveragePooling2D()(input_features)
    squeezed = Reshape((1,1,-1))(squeezed)
    squeezed = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(squeezed)
    squeezed = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(squeezed)
    
    excited = multiply([squeezed,input_features])
    return excited

#both 
def scSE(input_features, ratio = 2):
    _cse = sSE(input_features)
    _sse = cSE(input_features,ratio)
    return add([_cse,_sse])
