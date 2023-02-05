import pyspark
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StringIndexer


def correlation(dataset, ordinal_attributes, numeric_attributes, target_col = "Attrition"):
  '''
  Parameters:
    dataset: Pyspark sql dataframe
    ordinal_attributes: list of strings, name of ordinal attributes
    numeric_attributes: list of strings, name of numeric attributes
    target_col: name of label columns
  Return: 
    numpy array: correlation array
  '''

  indexed_name = [col_name + "Indexed" for col_name in ordinal_attributes]

  #transforming feature
  stringIndexer = StringIndexer(inputCols = ordinal_attributes + ['Attrition'], outputCols = indexed_name + ['label']).fit(dataset)
  assembler = VectorAssembler(
      inputCols= numeric_attributes + indexed_name + ['label'],
      outputCol="features")
  
  dataset = stringIndexer.transform(dataset)
  dataset = assembler.transform(dataset)

  corr_df = Correlation.corr(dataset= dataset, column= 'features', method= 'spearman')

  return corr_df.collect()[0][0].toArray()