import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].
    u_o - outer words vector - true_w
    v_c - center words - - inputs
     
    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """    
#     cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# 
#     y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
#     cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
#                              + (1 - y) * tf.log(1 - y_clipped), axis=1))

#     inputs = tf.clip_by_value(inputs, 1e-10, 0.9999999)
#     true_w = tf.clip_by_value(true_w, 1e-10, 0.9999999)
    print('inputs shape',inputs)
    
    
    A = tf.matmul(inputs, 
                  tf.matrix_transpose(true_w)
                )
 
     
    B =  tf.log(tf.add(1e-10,tf.reduce_sum(tf.exp(
                        tf.matmul(inputs,tf.matrix_transpose(true_w))
                        ),1)))
    

    AB =  tf.subtract(B, A)
    
#     ABin = tf.nn.softmax_cross_entropy_with_logits(logits = inputs, labels=true_w)
#     ABin = tf.Print(ABin, [ABin], message='DEBUG ABin: ')
#     print('-------------------')
#     print(tf.subtract(B, A))
#     print(tf.nn.softmax_cross_entropy_with_logits(logits = inputs, labels=true_w))
#     print('-------------------')
#     ABin = tf.Print(ABin, [ABin], message='DEBUG ABin: ')
#     AB = tf.Print(AB, [AB], message='DEBUG AB: ')
#     tf.Print(tf.subtract(B, A))
#     tf.Print(tf.nn.softmax_cross_entropy_with_logits(logits = inputs, labels=true_w))
    
    print('-------------------')
    return AB


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    
    print(inputs, 'inputs shape')
    print(weights, 'weights shape')
    print(biases, 'biases shape')
    print(labels, 'labels shape')
    print(sample.shape, 'sample shape')
    print(len(unigram_prob), 'unigram_prob shape')
    k = sample.shape[0]*1.0
    
    print("------")
    u_o = tf.nn.embedding_lookup(weights, labels)
    u_o = tf.reshape(u_o, [-1, tf.shape(inputs)[1]])
    print(u_o, "u_o shape")
    
    noise = tf.nn.embedding_lookup(weights, sample) 
    print(noise, 'noise shape')
      
    b_l= tf.nn.embedding_lookup(biases, labels)
    print(b_l, 'b_l shape')
    
    b_s= tf.nn.embedding_lookup(biases, sample)
    b_s = tf.reshape(b_s, [tf.shape(b_s)[0],1])
    print(b_s, 'b_s shape')
    print("------")
    
    unigram_prob_tensor = tf.convert_to_tensor(unigram_prob, dtype=tf.float32)
    
    s_o = tf.add(tf.matmul(inputs, tf.matrix_transpose(u_o)), tf.matrix_transpose(b_l))
    
    
    s_x = tf.add(tf.matmul(inputs, tf.matrix_transpose(noise)), tf.matrix_transpose(b_s))
    
#     pr_wo = tf.gather(unigram_prob,lable)
#     pr_wo = tf.Print(pr_wo, [pr_wo], message='DEBUG pr_wo: ')
# 
#     pr_wx = tf.gather(unigram_prob,sample)
#     pr_wx = tf.Print(pr_wx, [pr_wx], message='DEBUG pr_wx: ')
    
    pr_wo = tf.matrix_transpose(
        tf.nn.embedding_lookup(unigram_prob_tensor,labels))

    pr_wx = tf.nn.embedding_lookup(unigram_prob_tensor,sample)


    pr_wo_wc = tf.sigmoid(tf.subtract(s_o, tf.log(
        tf.add(1e-10,tf.scalar_mul(k,pr_wo))), name='PrD1'))
    
    pr_wx_wc = tf.sigmoid(tf.subtract(s_x, tf.log(
        tf.add(1e-10,tf.scalar_mul(k,pr_wx))), name='PrD0'))
    
     
    #pr_wx_wc_updated = tf.log(tf.add(1,tf.scalar_mul(-1,pr_wx_wc)))
#     pr_wo_wc = tf.Print(pr_wo_wc, [pr_wo_wc], message='DEBUG s_o: ')
#     pr_wx_wc = tf.Print(pr_wx_wc, [pr_wx_wc], message='DEBUG s_o: ')    
#     u_o  = tf.Print(u_o, [u_o], message='DEBUG os: ')
#     noise = tf.Print(noise, [noise], message='DEBUG noise: ')
#     b_l = tf.Print(b_l, [b_l], message='DEBUG b_l: ')
#     b_s = tf.Print(b_s, [b_s], message='DEBUG b_s: ')
#     s_o = tf.Print(s_o, [s_o], message='DEBUG s_o: ')
#     s_x = tf.Print(s_x, [s_x], message='DEBUG s_o: ')
#     pr_wo = tf.Print(pr_wo, [pr_wo], message='DEBUG pr_wo: ')
#     pr_wx = tf.Print(pr_wx, [pr_wx], message='DEBUG pr_wx: ')
    
    nceLoss = tf.scalar_mul(-1.0,
                    tf.add(
                        tf.log(tf.add(1e-10,pr_wo_wc)),
                        tf.reduce_sum(
                               tf.log(tf.add(1e-10,tf.subtract(1.0,pr_wx_wc))),1))
            )
    
    print(nceLoss)
    
#     nceLoss = tf.nn.nce_loss(weights,
#     biases,
#     labels,
#     inputs,
#     len(sample),
#     len(unigram_prob),
#     num_true=1,
#     sampled_values=None,
#     remove_accidental_hits=False,
#     partition_strategy='mod',
#     name='nce_loss'
#     )
    
    return nceLoss