from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, Evaluator
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier, LinearSVC, LogisticRegression


def trainModel(train_data, mlModel, paramGrid = None):
  #define categorical and numeric cols for StringIndexer
  target_col = "Attrition"
  variable_cols = []
  categorical_cols = []
  numeric_cols = []
  for col_name, col_type in train_data.dtypes:
    if col_name != target_col:
      variable_cols.append(col_name)
    if col_type != 'string':
      numeric_cols.append(col_name)    
    elif col_name != target_col:     
      categorical_cols.append(col_name)
  indexed_name = [col_name + "Indexed" for col_name in categorical_cols]

  #transforming feature
  stringIndexer = StringIndexer(inputCols = categorical_cols + ['Attrition'], outputCols = indexed_name + ['label'])
  assembler = VectorAssembler(
      inputCols= numeric_cols + indexed_name,
      outputCol="nonScaledFeatures")
  minMaxScaler = MinMaxScaler(inputCol = 'nonScaledFeatures', outputCol = 'features')

  #define machine learning pipeling
  pipeline = Pipeline(stages=[stringIndexer, assembler, minMaxScaler, mlModel])

  if paramGrid != None:
    #define cross validator, it will identify best param defined in paramGrid by k-fold cross validation
    #and refit train_data set with best param
    crossValidator = CrossValidator(estimator= pipeline,
                                    estimatorParamMaps= paramGrid,
                                    evaluator= BinaryClassificationEvaluator(),
                                    numFolds= 5)
    
    #fit validator to trainset
    bestModel = crossValidator.fit(train_data)
    
    return bestModel

  else:
    model = pipeline.fit(train_data)
    
    return model
