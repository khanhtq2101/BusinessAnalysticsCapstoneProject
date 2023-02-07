from eda.eda import correlation

import seaborn as sns
import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd
from pyspark.sql.functions import col


def correlation_heatmap(dataset, ordinal_attributes, numeric_attributes, target_col = "Attrition", masked= True):
  '''
  Parameters:
    dataset: Pyspark sql dataframe
    ordinal_attributes: list of strings, name of ordinal attributes
    numeric_attributes: list of strings, name of numeric attributes
    target_col: name of label columns
    masked: boolean, whethter mask the upper part of heatmap or not
  Return: 
    figure of heatmap
  '''
  corr_heatmap = correlation(dataset, ordinal_attributes, numeric_attributes, target_col)

  mask = np.zeros_like(corr_heatmap)
  if masked:
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if j >= i:
                mask[i, j] = 1
  fig = plt.figure()
  fig.set_size_inches(16, 14)
  ax = fig.add_axes([0, 0, 1, 1])

  sns.heatmap(corr_heatmap, axes = ax, cmap="YlGnBu", mask= mask, vmin= -0.9, vmax= 0.9, annot= True)

  ax.set_xticklabels(numeric_attributes + ordinal_attributes + ['Attrition'],
                    rotation=45,
                    horizontalalignment='right')
  ax.set_yticklabels(numeric_attributes + ordinal_attributes + ['Attrition'],
                    rotation= 0)
  
  return fig


def continous_plot(dataset, continous_attributes, target_name= "Attrition", plot_type= 'boxplot', grouped= True):
  '''
  Parameter:
    continous_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (boxplot, histplot, kdeplot, violinplot)
    grouped: boolean, whether group by target attribute or not
  Return: 
    figure of plot
  '''

  fig, axes = plt.subplots(len(continous_attributes) // 6, 6)
  fig.set_size_inches(20, 2 + 3*(len(continous_attributes) // 6))

  for (i, att_name) in enumerate(continous_attributes):
    selected_data = dataset.select([att_name, target_name]).collect()
    selected_data = pd.DataFrame(selected_data, columns= [att_name, target_name])
    axes[i // 6, i % 6].set_xlabel(att_name)

    if plot_type == 'boxplot':
      sns.boxplot(data= selected_data, x= att_name, y= target_name, ax = axes[i // 6, i % 6])
    elif plot_type == 'violinplot':
      sns.violinplot(data= selected_data, x= att_name, y= target_name, ax = axes[i // 6, i % 6])
    elif plot_type == 'kdeplot':
      sns.kdeplot(data= selected_data, x= att_name, hue= target_name, ax = axes[i // 6, i % 6])
    elif plot_type == 'histplot':
      if grouped:
        sns.histplot(data= selected_data, x= att_name, hue = target_name, 
                      multiple="dodge", ax = axes[i // 6, i % 6], kde=True)
      else:
        sns.histplot(data= selected_data, x= att_name, ax = axes[i // 6, i % 6], kde=True)
    if plot_type != 'boxplot':
      if i != 5:
        axes[i // 6, i % 6].get_legend().remove()
    if i % 6 != 0:
      axes[i // 6, i % 6].set_ylabel('')

  return fig


def hist_ordinal(dataset, ordinal_attributes, target_name= "Attrition", attrition_percent= True):
  '''
  Parameter:
    ordinal_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (countplot, histplot))
    attrition_percent: boolean plot the percentage of attrition line
  Return: 
    figure of plot
  '''

  fig, axes = plt.subplots(3, 4)
  fig.set_size_inches(21, 10)

  for (i, att_name) in enumerate(ordinal_attributes):
    selected_data = dataset.select([att_name, target_name]).collect()
    selected_data = pd.DataFrame(selected_data, columns= [att_name, target_name])
    if att_name == "BusinessTravel":
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i//4, i%4], order= ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])  
    else:
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i//4, i%4])
    if i != 3:
      axes[i//4, i%4].get_legend().remove()
    if i % 4 != 0:
      axes[i//4, i%4].set_ylabel('')

    #plot percentage line
    if attrition_percent:
      count = dataset.groupBy([att_name, target_name]).count().sort(att_name)
      values = np.array(count.select(att_name).distinct().sort(att_name).collect())
      att_values = [values[i, 0] for i in range(values.shape[0])]
      count_total = []
      count_yes = []
      count_no = []

      for value in att_values:
        n_yes = count.where((col(target_name) == 'Yes') & (col(att_name) == value)).select('count').collect()[0][0]
        n_no = count.where((col(target_name) == 'No') & (col(att_name) == value)).select('count').collect()[0][0]
        count_yes.append(n_yes)
        count_no.append(n_no)
        count_total.append(n_yes + n_no)

      yes_per = [(count_yes[i] / count_total[i])*100 for i in range(len(att_values))]
      
      twin = axes[i//4, i%4].twinx()
      twin.plot(list(range(len(yes_per))), yes_per, marker= 'o', color= 'cadetblue', label= "Attrition rate", linewidth= 2, markersize= 8)
      twin.set_yticks(list(range(0, 101, 20)))
      if i == 3:
        twin.legend()
      if i % 4 == 3:
        twin.set_ylabel('percentage')
  axes[2, -1].axis('off')


  fig.suptitle("Histogram of ordinal variables by attrition", size= 16)
  
  return fig


def nominal_plot(dataset, nominal_attributes, target_name= "Attrition", plot_type= 'countplot'):
  '''
  Parameter:
    nominal_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (countplot)
  Return: 
    figure of plot
  '''

  fig, axes = plt.subplots(1, len(nominal_attributes))
  fig.set_size_inches(18, 1 + 3*(len(nominal_attributes) // 5))

  for (i, att_name) in enumerate(nominal_attributes):
    selected_data = dataset.select([att_name, target_name]).collect()
    selected_data = pd.DataFrame(selected_data, columns= [att_name, target_name])
    axes[i].set_xlabel(att_name)

    if plot_type == 'countplot':
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i])
      axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
    if i >=1:
      axes[i-1].get_legend().remove()
      

  return fig


def percentage_stacked(dataset, nominal_attributes, target_name= "Attrition", plot_type= 'countplot'):
  '''
  Parameter:
    nominal_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (countplot, histplot))
  Return: 
    figure of plot
  '''  
  fig, axes = plt.subplots(1, len(nominal_attributes))
  fig.set_size_inches(20, 1 + 3*(len(nominal_attributes) // 5))

  for (i, att_name) in enumerate(nominal_attributes):
    count = dataset.groupBy([att_name, target_name]).count().sort(att_name)

    values = np.array(count.select(att_name).distinct().sort(att_name).collect())
    att_values = [values[i, 0] for i in range(values.shape[0])]
    count_yes = []
    count_no = []
    count_total = []

    for value in att_values:
      n_yes = count.where((col(target_name) == 'Yes') & (col(att_name) == value)).select('count').collect()[0][0]
      n_no = count.where((col(target_name) == 'No') & (col(att_name) == value)).select('count').collect()[0][0]
      count_yes.append(n_yes)
      count_no.append(n_no)
      count_total.append(n_yes + n_no)

    yes_per = [(count_yes[i] / count_total[i])*100 for i in range(len(att_values))]
    no_per = [100 - yes_per[i] for i in range(len(att_values))]
    axes[i].bar(att_values, yes_per, label = 'Yes')
    axes[i].bar(att_values, no_per, bottom= yes_per, label= 'No')
    axes[i].tick_params(axis='x', rotation=90)

    axes[i].set_xlabel(att_name)
    axes[i].set_ylabel("Percentage")
    if i != len(nominal_attributes):
      axes[i].set_ylabel('')

  axes[-1].legend()

  return fig