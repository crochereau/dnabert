import tensorflow as tf

def masked_sparse_ca(labels, logits):
    '''Sparse categorical accuracy for masked language modeling'''

    # make sure only labels that are not equal to -100 affect the loss
    active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
    active_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
    predictions = tf.math.argmax(active_logits, axis=-1)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    result = predictions == tf.cast(labels, tf.int64)
    return result


def masked_sparse_cce(labels, logits):
    '''Sparse categorical cross entropy loss for masked language modeling'''

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )
    # make sure only labels that are not equal to -100 affect the loss
    active_loss = tf.not_equal(tf.reshape(labels, (-1,)), -100)
    reduced_logits = tf.boolean_mask(tf.reshape(logits, (-1, shape_list(logits)[2])), active_loss)
    labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
    return loss_fn(labels, reduced_logits)


def shape_list(tensor):
    # function from Huggingface
    """
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        tensor (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    dynamic = tf.shape(tensor)

    if tensor.shape == tf.TensorShape(None):
        return dynamic

    static = tensor.shape.as_list()

    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

