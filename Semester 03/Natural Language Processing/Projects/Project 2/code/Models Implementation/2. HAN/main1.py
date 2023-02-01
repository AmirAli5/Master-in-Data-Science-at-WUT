from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from han import HAN

max_features = 5000
maxlen_sentence = 16
maxlen_word = 25
batch_size = 32
embedding_dims = 50
epochs = 10

df = pd.read_csv("Reviews.csv")
corpus = []
for i in range(0, len(df)):
    review = df['Text'][i]
    review = ''.join(review)
    corpus.append(review)
    
voc_size = 5000

onehot_repr=[one_hot(words,voc_size)for words in corpus] 

sent_length=25
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)


# target attributes
encoded_df = pd.get_dummies(df['Score'])
labels = np.array(encoded_df)

# split the dataset into train and test
X, Y = text, target_onehot
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print('Build model...')
model = HAN(maxlen_sentence, maxlen_word, max_features, embedding_dims)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

print('Train...')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=3, mode='max')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          callbacks=[early_stopping],
          validation_data=(x_test, y_test))

print('Test...')
result = model.predict(x_test)
