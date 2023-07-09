import os
import shutil
import pandas as pd
import tensorflow as tf
print(tf.__version__)
import tensorflow_hub as hub
print(hub.__version__)
from wordcloud import WordCloud
import tensorflow_text as text
# print(text.__version__)
#from official.nlp import optimization  # to create AdamW optimizer
from transformers import BertModel#, RobertaModel
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
import nltk
import re
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModel, TFAutoModel,TFRobertaForSequenceClassification, TFBertModel, TFRobertaModel, AutoConfig, TFAutoModelForSequenceClassification, BertTokenizer, BertModel
import preprocessor as pp
import emoji
import torch
import transformers
import itertools
import seaborn as sns
from nltk.util import ngrams
# import gensim
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from PIL import Image
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.optimizers import Adam, Adamax
import shap

nltk.download('punkt')
nltk.download('wordnet')
tf.get_logger().setLevel('ERROR')
nltk.download('stopwords')

stops_en = set(stopwords.words("english"))
stops_es = set(stopwords.words("spanish"))

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier

df_stops_es = pd.read_csv("México/stops_spanish.txt", sep=" ", header=None, encoding="utf-8")
for word in df_stops_es[0]:
  stops_es.add(word)
import re
# Comprueba que hay una GPU disponible y en caso afirmativo indica cual
def comprobarGPU():
  if torch.cuda.is_available():    

      # Tell PyTorch to use the GPU.    
      device = torch.device("cuda")

      print('There are %d GPU(s) available.' % torch.cuda.device_count())

      print('We will use the GPU:', torch.cuda.get_device_name(0))

  # If not...
  else:
      print('No GPU available, using the CPU instead.')
      device = torch.device("cpu")

def plot_confusion_matrixScikit(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.grid(False)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    #plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlabel('Predicted label')
    plt.show()


def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()



def cleantext(string, idioma="EN"):
    text = string.lower().split()
    text = " ".join(text)
    text = re.sub(r"http(\S)+",' ',text)    
    text = re.sub(r"www(\S)+",' ',text)
    text = re.sub(r"&",' and ',text)  
    text = text.replace('&amp',' ')
    text = re.sub(r"[^0-9a-zA-Z]+",' ',text)
    text = text.split()

    if idioma == "EN":
      text = [w for w in text if not w in stops_en]
    else:
      text = [w for w in text if not w in stops_es]

    text = " ".join(text)
    return text

# Revisar
def cleanhastag(text):
    # clean
    # text = text.replace("#","")
    # remove
    # print(text)
    clean_tweet = re.sub("@[A-Za-z0-9_]+","", text)
    # print(clean_tweet)
    clean_tweet = re.sub("#[A-Za-z0-9_]+","", clean_tweet)
    # print(clean_tweet)
    clean_tweet = re.sub(r'http\S+', '', clean_tweet)
    
    return clean_tweet.strip()


emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

# Para aplicar cualquiera de los preprocesados posibles
def preprocess(texts, lowercase, python_tokenize, demojize, clean_hashtag, cardiff, remove_emojis, clean_et=True):

  if lowercase:
    print("lowercase")
    texts = [text.lower() for text in texts]
  if python_tokenize:
    print("python_tokenize")
    texts = [pp.tokenize(text) for text in texts]
  if demojize:
    print("demojize")
    texts = [emoji.demojize(text) for text in texts]
  if clean_hashtag:
    print("clean_hashtag")
    texts = [cleanhastag(text) for text in texts]
  if cardiff:
    print("cardiff")
    texts = [preprocessCardiff(text) for text in texts]
  if remove_emojis:
    print("remove_emojis")
    texts = [emoji_pattern.sub(r'',text) for text in texts]
  if clean_et:
    print("clean_et")
    texts = [text.replace('&amp;','and') for text in texts]

  
  

  # Por si acaso lo paso a lista
  texts = [text for text in texts]

  return texts

def preprocessCardiff(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Función para tokenizar las palabras para que Bert pueda procesarlas
# Trunca, devuelve atención y añade los tokens especiales de BERT como
# SEP y CLS
def bert_encode(tokenizer,data,maximum_length) :
  input_ids = []
  attention_masks = []
  

  for i in range(len(data)):
      encoded = tokenizer.encode_plus(
        
        data[i],
        add_special_tokens=True,
        max_length=maximum_length,
        pad_to_max_length=True,
        truncation = True,
        return_attention_mask=True,
      )
      
      input_ids.append(encoded['input_ids'])
      attention_masks.append(encoded['attention_mask'])

  return np.array(input_ids),np.array(attention_masks)
  

# Devuelve la media, mediana y maximo de un conjunto de textos
def getMediaMedianaMaximoPalabras(texts):
  contador = []
  for text in texts:
    words = text.split()
    if len(words) < 304:
      contador.append(len(words))


  return np.mean(contador), np.median(contador), np.max(contador)

# Devuelve una lista con los tamaños de los textos
def getTextsSizes(texts):
  contador = []
  for text in texts:
    words = text.split()
    if len(words) < 304:
      contador.append(len(words))
      if len(words) == 304:
        print(len(contador))
        print(text)
    

  return contador

# Hago un map para poder entrenar mejor los datos
def mapFakesTrues(df_train, df_test):
  df_train['label'] = df_train['label'].map({'fake': 1, 'real': 0})
  df_test['label'] = df_test['label'].map({'fake': 1, 'real': 0})

# Separo el texto de las etiquetas
def getTextsLabelsIngles(df_train, df_test):
  train_texts = df_train['tweet']
  train_labels = df_train['label']
  test_texts = df_test.tweet.values
  test_labels = df_test['label']

  return train_texts, train_labels, test_texts, test_labels

def getTextsLabelsInglesNER(df_train, df_test):
  train_texts = df_train['tweet_ner']
  train_labels = df_train['label']
  test_texts = df_test.tweet_ner.values
  test_labels = df_test['label']

  return train_texts, train_labels, test_texts, test_labels


def cargarDatasetIngles():
  df_train = pd.read_excel('Contraint@AAAI/Constraint_English_Train.xlsx')
  df_val = pd.read_excel('Contraint@AAAI/Constraint_English_Val.xlsx')
  df_test = pd.read_excel('Contraint@AAAI/english_test_with_labels.xlsx')

  df_train = pd.concat([df_train, df_val])

  mapFakesTrues(df_train,df_test)

  return df_train, df_test

def cargarDatasetMexico():
  df_train = pd.read_excel('México/train.xlsx')
  df_test = pd.read_excel('México/test.xlsx')

  df_train['Category'] = df_train['Category'].map({'Fake': 1, 'True': 0})

  return df_train, df_test


def getTextsLabelsMexico(df_train, df_test):
  train_texts = df_train['Text']
  train_labels = df_train['Category']
  test_texts = df_test.TEXT.values
  test_labels = df_test['CATEGORY']

  return train_texts, train_labels, test_texts, test_labels


def plot_top_ngrams_barchart(text, n=2, http=False, color="Greens_d", idioma="EN"):

    if idioma == "EN":
      stop = set(stopwords.words('english'))
    else:
      stop = set(stopwords.words('spanish'))
      stop.add("number")

    if http:
      stop.add('http')
      stop.add('https')
      stop.add('co')
      stop.add('URL')
      stop.add('url')



    # new = text.str.split()
    # new = new.values.tolist()
    corpus=[word for i in text for word in i]


    def _get_top_ngram(corpus, n=None):
        vec = CountVectorizer(ngram_range=(n, n), stop_words=stop).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) 
                      for word, idx in vec.vocabulary_.items()]
        words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:10]

    top_n_bigrams=_get_top_ngram(text,n)[:10]
    x,y=map(list,zip(*top_n_bigrams))
    pal = sns.color_palette(color, 10)
    fig = sns.barplot(x=y,y=x, palette=pal[::-1])
    return fig

def makeClouds(df, idioma, colormap="viridis"):

  if idioma == "EN":
    stop = stops_en
    stop.add('http')
    stop.add('https')
    stop.add('co')
    stop.add('URL')
    stop.add('url')
  else:
    stop = stops_es
    stop.add('http')
    stop.add('https')
    stop.add('co')
    stop.add('URL')
    stop.add('url')
    stop.add("number")
    

  true_words = []
  for sentence in df:
    for word in sentence.split():
      true_words.append(word)

  true_words = ' '.join(true_words)

  cloud_true = WordCloud(stopwords = stop,background_color="white", max_words=400, width=500, height=500, colormap=colormap).generate(true_words)

  return cloud_true

def get_lda_objects(text):
    nltk.download('stopwords')    
    stop=set(stopwords.words('english'))
    stop.add('http')
    stop.add('https')
    stop.add('co')
    stop.add('URL')
    stop.add('url')

    
    def _preprocess_text(text):
        corpus=[]
        stem=PorterStemmer()
        lem=WordNetLemmatizer()
        for news in text:
            words=[w for w in word_tokenize(news) if (w not in stop)]

            words=[lem.lemmatize(w) for w in words if len(w)>2]

            corpus.append(words)
        return corpus
    
    corpus=_preprocess_text(text)
    
    dic=gensim.corpora.Dictionary(corpus)
    bow_corpus = [dic.doc2bow(doc) for doc in corpus]
    
    lda_model =  gensimvis.LdaMulticore(bow_corpus, 
                                   num_topics = 4, 
                                   id2word = dic,                                    
                                   passes = 10,
                                   workers = 2)
    
    return lda_model, bow_corpus, dic

def plot_lda_vis(lda_model, bow_corpus, dic):
    pyLDAvis.enable_notebook()
    vis = gensimvis.prepare(lda_model, bow_corpus, dic)
    return vis

def EDAIngles(train_texts):
  # N° de Items
  sizes = getTextsSizes(train_texts)
  print(f"Tamaño total del conjunto de datos: {len(df_train) + len(df_test)}")
  print(f"N° de palabras del conjunto de entrenamiento: {np.sum(sizes)}")
  print(f"Tamaño del conjunto de datos de entrenamiento: {len(df_train)}")
  print(f"Tamaño del conjunto de datos de test: {len(df_test)}")
  # Media Mediana y Maximo de palabras (del train, para que no haya data snoping)
  media, mediana, maximo = getMediaMedianaMaximoPalabras(train_texts) # Revisar: Hago los mismo con las letras/caracteres
  print(f"Media del número de palabras: {media}")
  print(f"Mediana del número de palabras: {mediana}")
  print(f"Máximo del número de palabras: {maximo}")

  print(f"Gráfica con la distribución del número de palabras en el conjunto de entrenamiento:")
  # Use the seborn style
  plt.style.use('seaborn')
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif"})
  
  fig = sns.histplot(data=sizes, binwidth=3)
  fig.set_xlabel("N° palabras")
  fig.set_title("Distribución del número de palabras en el conjunto de entrenamiento")
  plt.show()

  # Unigramas, bigramas, trigramas

  
  sns.set(rc={'figure.figsize':(10,5)})
  # Unigrams
  plt.figure(figsize=(10,5))
  fig1 = plot_top_ngrams_barchart(train_texts, n=1, color="Greens_d", idioma="EN")
  fig1.set_xlabel("N° de repeticiones")
  fig1.set_title("10 Unigramas más repetidos")
  # plt.set_size_inches(10, 5)
  # plt.show()
  plt.savefig("Unigramas.png",bbox_inches='tight')

  # Bigrams
  plt.figure(figsize=(10,5))
  fig2 = plot_top_ngrams_barchart(train_texts, n=2, color="Blues_d", idioma="EN")
  fig2.set_xlabel("N° de repeticiones")
  fig2.set_title("10 Bigramas más repetidos")
  plt.figure(figsize=(10,5))
  # plt.set_size_inches(10, 5)
  plt.savefig("Bigramas.png",bbox_inches='tight')

  #Trigrams
  plt.figure(figsize=(10,5))
  fig3 = plot_top_ngrams_barchart(train_texts, n=3, color="Reds_d", idioma="EN")
  fig3.set_xlabel("N° de repeticiones")
  fig3.set_title("10 Trigramas más repetidos")
  plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Trigramas.png",bbox_inches='tight')

  # Lo repetimos sin https co

  # Unigrams
  plt.figure(figsize=(10,5))
  fig1 = plot_top_ngrams_barchart(train_texts, n=1, http=True, color = "Greens_d", idioma="EN")
  fig1.set_xlabel("N° de repeticiones")
  fig1.set_title("10 Unigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Unigramas sin HTTP.png",bbox_inches='tight')

  # Bigrams
  plt.figure(figsize=(10,5))
  fig2 = plot_top_ngrams_barchart(train_texts, n=2, http=True, color = "Blues_d", idioma="EN")
  fig2.set_xlabel("N° de repeticiones")
  fig2.set_title("10 Bigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Bigramas sin HTTP.png",bbox_inches='tight')

  #Trigrams
  plt.figure(figsize=(10,5))
  fig3 = plot_top_ngrams_barchart(train_texts, n=3, http=True, color = "Reds_d", idioma="EN")
  fig3.set_xlabel("N° de repeticiones")
  fig3.set_title("10 Trigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Trigramas sin HTTP.png",bbox_inches='tight')

  # Word count
  print("Nubes de palabras del conjunto de entrenamiento")
  cloud = makeClouds(train_texts, idioma="EN")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras del conjunto de entrenamiento")
  plt.savefig("Wordcloud.png")


  # Revisar: Añadir los N-gramas  separando fakes y news
  # Separo True and Fake
  fakes = df_train.loc[df_train['label'] == 1]
  trues = df_train.loc[df_train['label'] == 0]

  fakes = preprocess(fakes.tweet, lowercase=True, python_tokenize=False, demojize=False, clean_hashtag=True, cardiff=False, remove_emojis=False)
  trues = preprocess(trues.tweet, lowercase=True, python_tokenize=False, demojize=False, clean_hashtag=True, cardiff=False, remove_emojis=False)

  # Trues
  cloud = makeClouds(trues, idioma="EN")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras de TRUES")
  plt.savefig("Wordcloud.png")

  # Fakes
  cloud = makeClouds(fakes, idioma="EN")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras de FAKES")
  plt.savefig("Wordcloud.png")

  # Ver si uso LDA
  # lda_model, bow_corpus, dic = get_lda_objects(train_texts)
  # lda_model.show_topics()
  # plot_lda_vis(lda_model, bow_corpus, dic)


def EDAMexico():
  # N° de Items
  sizes = getTextsSizes(train_texts)
  print(f"Tamaño total del conjunto de datos: {len(df_train) + len(df_test)}")
  print(f"N° de palabras del conjunto de entrenamiento: {np.sum(sizes)}")
  print(f"Tamaño del conjunto de datos de entrenamiento: {len(df_train)}")
  print(f"Tamaño del conjunto de datos de test: {len(df_test)}")
  # Media Mediana y Maximo de palabras (del train, para que no haya data snoping)
  media, mediana, maximo = getMediaMedianaMaximoPalabras(train_texts) # Revisar: Hago los mismo con las letras/caracteres
  print(f"Media del número de palabras: {media}")
  print(f"Mediana del número de palabras: {mediana}")
  print(f"Máximo del número de palabras: {maximo}")

  print(f"Gráfica con la distribución del número de palabras en el conjunto de entrenamiento:")
  # Use the seborn style
  plt.style.use('seaborn')
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif",
      # "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })
  
  fig = sns.histplot(data=sizes, binwidth=3)
  fig.set_xlabel("N° palabras")
  fig.set_title("Distribución del número de palabras en el conjunto de entrenamiento")
  plt.show()

  # Unigramas, bigramas, trigramas

  
  sns.set(rc={'figure.figsize':(10,5)})
  # Unigrams
  plt.figure(figsize=(10,5))
  fig1 = plot_top_ngrams_barchart(train_texts, n=1, color="Greens_d", idioma="ES")
  fig1.set_xlabel("N° de repeticiones")
  fig1.set_title("10 Unigramas más repetidos")
  # plt.set_size_inches(10, 5)
  # plt.show()
  plt.savefig("Unigramas.png",bbox_inches='tight')

  # Bigrams
  plt.figure(figsize=(10,5))
  fig2 = plot_top_ngrams_barchart(train_texts, n=2, color="Blues_d", idioma="ES")
  fig2.set_xlabel("N° de repeticiones")
  fig2.set_title("10 Bigramas más repetidos")
  plt.figure(figsize=(10,5))
  # plt.set_size_inches(10, 5)
  plt.savefig("Bigramas.png",bbox_inches='tight')

  #Trigrams
  plt.figure(figsize=(10,5))
  fig3 = plot_top_ngrams_barchart(train_texts, n=3, color="Reds_d", idioma="ES")
  fig3.set_xlabel("N° de repeticiones")
  fig3.set_title("10 Trigramas más repetidos")
  plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Trigramas.png",bbox_inches='tight')

  # Lo repetimos sin https co

  # Unigrams
  plt.figure(figsize=(10,5))
  fig1 = plot_top_ngrams_barchart(train_texts, n=1, http=True, color = "Greens_d", idioma="ES")
  fig1.set_xlabel("N° de repeticiones")
  fig1.set_title("10 Unigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Unigramas sin HTTP.png",bbox_inches='tight')

  # Bigrams
  plt.figure(figsize=(10,5))
  fig2 = plot_top_ngrams_barchart(train_texts, n=2, http=True, color = "Blues_d", idioma="ES")
  fig2.set_xlabel("N° de repeticiones")
  fig2.set_title("10 Bigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Bigramas sin HTTP.png",bbox_inches='tight')

  #Trigrams
  plt.figure(figsize=(10,5))
  fig3 = plot_top_ngrams_barchart(train_texts, n=3, http=True, color = "Reds_d", idioma="ES")
  fig3.set_xlabel("N° de repeticiones")
  fig3.set_title("10 Trigramas más repetidos (sin identificadores de URL)")
  # plt.figure(figsize=(10,5))
  # plt.show()
  plt.savefig("Trigramas sin HTTP.png",bbox_inches='tight')

  # Word count
  print("Nubes de palabras del conjunto de entrenamiento")
  cloud = makeClouds(train_texts, idioma="ES", colormap="inferno")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras del conjunto de entrenamiento")
  plt.savefig("Wordcloud.png")


  # Revisar: Añadir los N-gramas  separando fakes y news
  # Separo True and Fake
  fakes = df_train.loc[df_train['Category'] == 1]
  trues = df_train.loc[df_train['Category'] == 0]

  # fakes = preprocess(fakes.Text, lowercase=True, python_tokenize=True, demojize=False, clean_hashtag=True, cardiff=False, remove_emojis=False)
  # trues = preprocess(trues.Text, lowercase=True, python_tokenize=True, demojize=False, clean_hashtag=True, cardiff=False, remove_emojis=False)
  fakes = preprocess(fakes.Text, lowercase=False, python_tokenize=False, demojize=False, clean_hashtag=False, cardiff=False, remove_emojis=False)
  trues  = preprocess(trues.Text, lowercase=False, python_tokenize=False, demojize=False, clean_hashtag=False, cardiff=False, remove_emojis=False)
  # Trues
  cloud = makeClouds(trues, idioma="ES", colormap="inferno")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras de TRUES")
  plt.savefig("Wordcloud.png")

  # Fakes
  cloud = makeClouds(fakes, idioma="ES", colormap="inferno")
  plt.figure(figsize=(700/80,700/80))
  plt.imshow(cloud, interpolation="bilinear")
  plt.axis("off")
  plt.title("Nube de palabras de FAKES")
  plt.savefig("Wordcloud.png")

  # Ver si uso LDA
  # lda_model, bow_corpus, dic = get_lda_objects(train_texts)
  # lda_model.show_topics()
  # plot_lda_vis(lda_model, bow_corpus, dic)

def getTokenizerAndModel(model_name, model_normalization=False, from_pt = False, regularization=False):
  tokenizer = AutoTokenizer.from_pretrained(model_name, normalization=model_normalization)
  configuration = AutoConfig.from_pretrained(model_name)
  if regularization == True:
    configuration.attention_probs_dropout_prob = 0.5
    configuration.hidden_dropout_prob = 0.2

  model = TFAutoModel.from_pretrained(model_name,from_pt=from_pt,config=configuration)

  return tokenizer, model

# Palabras a añadir
def addPalabras():
  to_add = ["covid", "covid-19", "covid19", "coronavirus", "indiafightcorona", "lockdown", "COVID", "COVID-19"]
  tokenizer.add_tokens(to_add)

  # Compruebo que se han agregado correctamento
  for word in to_add:
    print(tokenizer.encode_plus(word))


def getCompleteReportScikitIngles(model, name,test_texts, digits=4):
  pred = model.predict(test_texts)
  pred = np.round(pred)
  pred = pred.flatten()

  # Use the seborn style
  # plt.style.use('seaborn')
  
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif"})
  

  print("------------- Classification Report -------------")
  print(classification_report(df_test['label'], pred, digits=5))
  plot_confusion_matrixScikit(confusion_matrix(df_test['label'],pred),target_names=['fake','real'], normalize = False, \
                      title = f'Confusion matix of {name} on test data')
  
  return pred

def getCompleteReportScikitMexico(model, name,test_texts, digits=4):
  pred = model.predict(test_texts)
  pred = np.round(pred)
  pred = pred.flatten()

  # Use the seborn style
  plt.style.use('seaborn')
  
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif"})
  

  print("------------- Classification Report -------------")
  print(classification_report(df_test['CATEGORY'], pred, digits=5))
  plot_confusion_matrixScikit(confusion_matrix(df_test['CATEGORY'],pred),target_names=['fake','real'], normalize = False, \
                      title = f'Confusion matix of {name} on test data')
  
  return pred

def getCompleteReportIngles(model, df_test, test_input_ids, test_attention_masks, digits=4):
  pred = model.predict([test_input_ids,test_attention_masks])
  pred = np.round(pred)
  pred = pred.flatten()

  # Use the seborn style
  plt.style.use('seaborn')
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif"})

  print("------------- Classification Report -------------")
  print(classification_report(df_test['label'], pred, digits=5))

  plot_confusion_matrixScikit(confusion_matrix(df_test['label'],pred),target_names=['fake','real'], normalize = False, \
                      title = f'Confusion matix of {model.name} on test data')
    
def getCompleteReportInglesSHAP(model, df_test, test_input_ids, test_attention_masks, digits=4):
  pred = model.predict([test_input_ids,test_attention_masks])
  # pred = np.round(pred)
  # pred = pred.flatten()
  pred = np.argmax(pred,axis=1)

  # Use the seborn style
  plt.style.use('seaborn')
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif"})

  print("------------- Classification Report -------------")
  print(classification_report(df_test['label'], pred, digits=5))

  plot_confusion_matrixScikit(confusion_matrix(df_test['label'],pred),target_names=['fake','real'], normalize = False, \
                      title = f'Confusion matix of {model.name} on test data')
  

def getCompleteReportMexico(model, df_test, test_input_ids, test_attention_masks, digits=4):
  pred = model.predict([test_input_ids,test_attention_masks])
  pred = np.round(pred)
  pred = pred.flatten()

  # Use the seborn style
  plt.style.use('seaborn')
  # But with fonts from the document body
  plt.rcParams.update({
      "font.family": "serif",
      # "text.usetex": True,     # use inline math for ticks
    "pgf.rcfonts": False     # don't setup fonts from rc parameters
    })

  print("------------- Classification Report -------------")
  print(classification_report(df_test['CATEGORY'], pred, digits=5))

  plot_confusion_matrixScikit(confusion_matrix(df_test['CATEGORY'],pred),target_names=['false','true'], normalize = False, \
                      title = f'Confusion matix of {model.name} on test data')

  return pred

def create_model_roberta(bert_model, model_name , lr, epsilon, optimizer,loss,sentence_length):
  input_ids = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  attention_masks = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  
  output = bert_model.roberta([input_ids,attention_masks])
  print(bert_model)
  # print(output[2])
  output = output[1]
  # output = tf.keras.layers.Dropout(0.5)(output)
 

  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  model = tf.keras.models.Model(name = model_name, inputs = [input_ids,attention_masks],outputs = output)

  model.compile(optimizer(learning_rate=lr,epsilon=epsilon), loss=loss, metrics=['accuracy'])
  return model

def create_model_bert(bert_model, model_name , lr, epsilon, optimizer,loss,sentence_length):
  input_ids = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  attention_masks = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  
  output = bert_model.bert([input_ids,attention_masks])
  # print(output)
  # print(output[2])
  output = output[1]
  # output = tf.keras.layers.Dropout(0.5)(output)
 

  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
  model = tf.keras.models.Model(name = model_name, inputs = [input_ids,attention_masks],outputs = output)

  model.compile(optimizer(learning_rate=lr,epsilon=epsilon), loss=loss, metrics=['accuracy'])
  return model

def create_model_bertSHAP(bert_model, model_name , lr, epsilon, optimizer,loss,sentence_length):
  input_ids = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  attention_masks = tf.keras.Input(shape=(sentence_length,),dtype='int32')
  
  output = bert_model.bert([input_ids,attention_masks])
  # print(output)
  # print(output[2])
  output = output[1]
  # output = tf.keras.layers.Dropout(0.5)(output)
 

  output = tf.keras.layers.Dense(2,activation='softmax')(output)
  model = tf.keras.models.Model(name = model_name, inputs = [input_ids,attention_masks],outputs = output)

  model.compile(optimizer(learning_rate=lr,epsilon=epsilon), loss=loss, metrics=['accuracy'])
  return model

def freezeLayers(model, unfreeze = False):
  for layer in model.layers:
    if 'tf_roberta_model' in layer.name:
      layer.trainable = False
    if 'tf_bert_model' in layer.name:
      layer.trainable = False

def trainModel(model,train_input_ids,train_attention_masks, train_labels, validation_split, epochs, batch_size, callbacks):
  history = model.fit(x=[train_input_ids,train_attention_masks],
                      y=train_labels,
                      validation_split=validation_split, 
                      epochs=epochs,
                      batch_size=batch_size,
                      callbacks=callbacks)
  return history

# Para serializar el modelo con pickle
def saveModel(model, path_name= "/content/drive/MyDrive/Colab Notebooks"):
  pickle.dump(open(path_name + "/" + model.name,"wb"))

# Para cargar el modelo serializado con pickle
def loadModel(model_name,path_name = "/content/drive/MyDrive/Colab Notebooks"):
  return pickle.load(open(path_name + "/" + model.name,"rb"))

def getExcelErrores(model_name, pred):
  val_ori = pd.read_excel('Contraint@AAAI/english_test_with_labels.xlsx')
  df_test = pd.read_excel('Contraint@AAAI/english_test_with_labels.xlsx')
  df_test['label'] = df_test['label'].map({'fake': 1.0, 'real': 0.0})
  svm_test_misclass_df = val_ori[pred!=df_test['label']]
  # Le doy la vuelta
  label_predicted = svm_test_misclass_df['label'].map({'fake': 'real', 'real': 'fake'})
  svm_test_misclass_df['label_predicted'] = label_predicted

  # Inserto los tweets ya procesados
  indices = svm_test_misclass_df.index
  test_texts_np = np.array(test_texts)
  svm_test_misclass_df.insert(3, 'tweet_processed', test_texts_np[indices])

  # Genero el excel
  svm_test_misclass_df.to_excel(f"Errores {model_name}.xlsx")

def getModelOutputs(x):
    a,b = bert_encode(tokenizer, x, 110)
    outputs = model_new([a,b])
    # print("Outputs:", outputs)
    # scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    # val = sp.special.logit(scores[:,1])
    # print("Val: ", val)
    return outputs

# explainer = shap.Explainer(miFNew,tokenizer,output_names=["Real", "Fake"])

# all_test_shap_values1000NEW = explainer(random.sample(test_texts,1000))
# pickle.dump(all_test_shap_values1000NEW,open("all_test_shap_values1000NEW.pkl","wb"))