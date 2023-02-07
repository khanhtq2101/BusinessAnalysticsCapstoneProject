from sklearn.metrics import confusion_matrix, precision_score, recall_score
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, Evaluator

def evaluateModel(test, mlModel):
  #get test prediction
  prediction = mlModel.transform(test)

  #calculate accuracy
  evaluator= MulticlassClassificationEvaluator(predictionCol="prediction")
  acc = evaluator.evaluate(prediction)

  #confusion matrix
  y_pred = prediction.select("prediction").collect()
  y_orig = prediction.select("label").collect()
  confusionMatrix = confusion_matrix(y_orig, y_pred)
  precision = precision_score(y_orig, y_pred, average= None)
  recall = recall_score(y_orig, y_pred, average= None)

  print("Accuracy:", acc)
  print("Confusion matrix:\n", confusionMatrix)
  print("Precision:", precision)
  print("Recall:", recall)

  return acc, confusionMatrix, precision, recall