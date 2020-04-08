import keras.backend as K
def ExactAUC(label_arg, pred_arg, weight = None):
  N = K.tf.size(label_arg, name="N")
  y_true = K.reshape(label_arg, shape=(N,))
  y_pred = K.reshape(pred_arg, shape=(N,))
  if weight is None:
    weight = K.tf.fill(K.shape(y_pred), 1.0)
  sort_result = K.tf.nn.top_k(y_pred, N, sorted=False, name="sort")
  y = K.gather(y_true, sort_result.indices)
  y_hat = K.gather(y_pred, sort_result.indices)
  w = K.gather(weight, sort_result.indices)
  is_negative = K.equal(y, K.tf.constant(0.0))
  is_positive = K.equal(y, K.tf.constant(1.0))
  w_zero = K.tf.fill(K.shape(y_pred), 0.0)
  w_negative = K.tf.where(is_positive, w_zero, w, name="w_negative")
  w_positive = K.tf.where(is_negative, w_zero, w)
  cum_positive = K.cumsum(w_positive)
  cum_negative = K.cumsum(w_negative)
  is_diff = K.not_equal(y_hat[:-1], y_hat[1:])
  is_end = K.tf.concat([is_diff, K.tf.constant([True])], 0)
  total_positive = cum_positive[-1]
  total_negative = cum_negative[-1]
  TP = K.tf.concat([
    K.tf.constant([0.]),
    K.tf.boolean_mask(cum_positive, is_end),
    ], 0)
  FP = K.tf.concat([
    K.tf.constant([0.]),
    K.tf.boolean_mask(cum_negative, is_end),
    ], 0)
  FPR = FP / total_negative
  TPR = TP / total_positive
  return K.sum((FPR[1:]-FPR[:-1])*(TPR[:-1]+TPR[1:])/2)

if __name__ == "__main__":
    test_dict = {
      "1 tie":{
        "label":[0, 0, 1, 1],
        "pred":[1.0, 2, 3, 1],
        "auc": 5/8,
      },
      "no ties, perfect":{
        "label":[0,0,1,1],
        "pred":[1,2,3,4],
        "auc":1,
      },
      "one bad error":{
        "label":[0,0,1,1],
        "pred":[1,2,3,-1],
        "auc":1/2,
      },
      "one not so bad error":{
        "label":[0,0,1,1],
        "pred":[1,2,3,1.5],
        "auc":3/4,
      }
    }
    sess = K.tf.Session()
    for test_name, test_data in test_dict.items():
      tensor_data = {
        k:K.tf.constant(v, K.tf.float32)
        for k,v in test_data.items()
      }
      g = ExactAUC(tensor_data["label"], tensor_data["pred"])
      auc = sess.run(g)
      if auc != test_data["auc"]:
        print("%s expected=%f computed=%f" % (test_name, test_data["auc"], auc))
