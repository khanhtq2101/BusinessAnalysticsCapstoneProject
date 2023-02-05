from eda.eda import correlation

import seaborn as sns
import numpy as np
import matplotlib.pyplot  as plt
import pandas as pd


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

  mask = np.ones_like(corr_heatmap)
  if masked:
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if j < i:
                mask[i, j] = 0
  fig = plt.figure()
  fig.set_size_inches(16, 12)
  ax = fig.add_axes([0, 0, 1, 1])

  sns.heatmap(corr_heatmap, axes = ax, cmap="YlGnBu", mask= mask, vmin= -0.9, vmax= 0.9, annot= True)

  ax.set_xticklabels(numeric_attributes + ordinal_attributes + ['Attrition'],
                    rotation=45,
                    horizontalalignment='right')
  ax.set_yticklabels(numeric_attributes + ordinal_attributes + ['Attrition'],
                    rotation= 0)
  
  return fig


def continous_plot(dataset, continous_attributes, target_name= "Attrition", plot_type= 'boxplot'):
  '''
  Parameter:
    continous_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (boxplot, histplot, kdeplot, violinplot)
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
      sns.histplot(data= selected_data, x= att_name, hue = target_name, 
                    multiple="dodge", ax = axes[i // 6, i % 6], kde=True)
      
  return fig

def hist_ordinal(dataset, ordinal_attributes, target_name= "Attrition"):
  fig, axes = plt.subplots(3, 4)
  fig.set_size_inches(20, 10)

  for (i, att_name) in enumerate(ordinal_attributes):
    selected_data = dataset.select([att_name, target_name]).collect()
    selected_data = pd.DataFrame(selected_data, columns= [att_name, target_name])
    if att_name == "BusinessTravel":
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i//4, i%4], order= ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])  
    else:
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i//4, i%4])
    if i != 3:
      axes[i//4, i%4].get_legend().remove()

  fig.suptitle("Histogram of ordinal variables by attrition", size= 16)
  return fig

def categorical_plot(dataset, categorical_attributes, target_name= "Attrition", plot_type= 'countplot'):
  '''
  Parameter:
    cetegorical_attributes: list of strings, name of attribute
    plot_type: type of plot, suppoted (countplot, histplot))
  Return: 
    figure of plot
  '''

  fig, axes = plt.subplots(1, len(categorical_attributes))
  fig.set_size_inches(18, 1 + 3*(len(categorical_attributes) // 5))

  for (i, att_name) in enumerate(categorical_attributes):
    selected_data = dataset.select([att_name, target_name]).collect()
    selected_data = pd.DataFrame(selected_data, columns= [att_name, target_name])
    axes[i].set_xlabel(att_name)

    if plot_type == 'countplot':
      sns.countplot(data= selected_data, x= att_name, hue = target_name, ax = axes[i])
      axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, horizontalalignment='right')
      if i >=1:
        axes[i-1].get_legend().remove()
    elif plot_type == 'histplot':
      if i == len(categorical_attributes) - 1:
        sns.histplot(data= selected_data, x= att_name, y = target_name, ax = axes[i], cbar= True)
      else: 
        sns.histplot(data= selected_data, x= att_name, y = target_name, ax = axes[i], cbar= False)
      axes[i].tick_params(axis='x', rotation=90)

  return fig