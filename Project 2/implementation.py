import tensorflow as tf
import re

BATCH_SIZE = 256
MAX_WORDS_IN_REVIEW = 150  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    data = review.split(" ")
    processed_review = []
    for word in data:
        word = word.lower() #preprocessing 1: lowercase
        #preprocessing 2: strip punctuation
        word = re.sub('[\s+\.\!\/_,$%^*<>(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+', '', word)
        word = re.sub('\d', '', word)
        if word not in stop_words and word !='' : #preprocessing 3: remove stop_words
            processed_review.append(word)

    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    #parameters
    learning_rate = 0.01
    n_class = 2
    n_hidden_units = 40  #128

    #input
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name= "input_data")
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, n_class], name="labels")
    #deal over-fitting
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name="dropout_keep_prob")
    #weights
    weights = tf.Variable(tf.random_normal(shape=[n_hidden_units, n_class], stddev=0.01))
    #bias
    bias = tf.Variable(tf.zeros([n_class]) + 0.1)
    #RNN cell
    #X_input = tf.reshape(input_data,[n_hidden_units, BATCH_SIZE, MAX_WORDS_IN_REVIEW])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units)
    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(cell=lstm_cell, output_keep_prob = dropout_keep_prob)
    #output
    output, state = tf.nn.dynamic_rnn(rnn_cell, input_data, dtype=tf.float32)
    output = tf.transpose(output, [1,0,2])
    #get the last output
    final_output = tf.gather(output, int(output.get_shape()[0]) - 1)
    #connect a hidden layer
    logits = tf.matmul(final_output, weights) + bias
    preds = tf.nn.softmax(logits)
    #compare the truth with the prediction
    compare_truth_pre = tf.equal(tf.argmax(preds,1), tf.argmax(labels,1))
    Accuracy = tf.reduce_mean(tf.cast(compare_truth_pre, dtype=tf.float32), name="accuracy")
    batch_xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = logits)
    loss = tf.reduce_mean(batch_xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
