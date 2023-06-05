import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.stats import pointbiserialr


app = typer.Typer(add_completion=False)


def get_percentage(comments):

  # Calculate the alignment percentage of all comments
  alignment_percentage = comments['Aligned_Value'].value_counts(normalize=True) * 100
  print(f"The value alignment percentage for all comments:", alignment_percentage)
  
  # Calculate the alignment percentage of delta/non-delta comments
  delta = comments[comments['delta_awarded'] == True]
  delta_alignment_percentage = delta['Aligned_Value'].value_counts(normalize=True) * 100
  print(f"The value alignment percentage for delta comments:", delta_alignment_percentage)

  non_delta = comments[comments['delta_awarded'] == False]
  non_delta_alignment_percentage = non_delta['Aligned_Value'].value_counts(normalize=True) * 100
  print("The value alignment percentage for non-delta comments:", non_delta_alignment_percentage)
  
  return delta_alignment_percentage, non_delta_alignment_percentage

@app.command()
def main(
    results_file: str = typer.Option(None, help='path for inference file')
    ):
  
  # Read data and modify the category of 'delta_awarded'
  data = pd.read_csv(results_file)
  data['delta_awarded'] = data['delta_awarded'].replace({'True': True, True: True, 'False': False, False: False})
  data = data[(data['delta_awarded']==True) | (data['delta_awarded']==False)]

  # Assgin human value to each post and comment
  columns_subset = ['Benevolence: dependability',
                    'Achievement', 'Benevolence: caring', 'Hedonism',
                    'Universalism: objectivity', 'Humility', 'Security: societal',
                    'Conformity: interpersonal', 'Self-direction: thought',
                    'Power: resources', 'Face', 'Power: dominance', 'Universalism: nature',
                    'Universalism: tolerance', 'Stimulation', 'Security: personal',
                    'Tradition', 'Universalism: concern', 'Conformity: rules',
                    'Self-direction: action']
  data['Human_Value'] = data[columns_subset].idxmax(axis=1)

  # Check value alignment of each comment and its post
  data['Aligned_Value'] = data['Human_Value'] == data.set_index('id')['Human_Value'].reindex(data['parent_id']).values
  comments = data[~(data["parent_id"].isnull())]

  # Calculate the alignment percentage of comments
  delta_alignment_percentage, non_delta_alignment_percentage = get_percentage(comments)
  
  # plot a stacked bar chart to show the distribution
  df = pd.DataFrame({'Aligned': [delta_alignment_percentage.get(True, 0), non_delta_alignment_percentage.get(True, 0)], 
                    'Non-aligned': [delta_alignment_percentage.get(False, 0), non_delta_alignment_percentage.get(False, 0)]})
  ax = df.plot(kind='bar', stacked=True)
  ax.set_title('Distribution of Value Alignment over Delta and Non-delta Comments')
  ax.set_xlabel('Delta Awarded')
  ax.set_xticklabels(['Delta', 'Non-delta'], rotation=0)
  ax.set_ylabel('Percentage')
  plt.legend(title='Value Alignment',loc = 'center', bbox_to_anchor=(1.2, 0.5))
  plt.show()

  # Save the figure as a PNG file
  plt.savefig('alignment_delta_distribution.png', dpi=300, bbox_inches='tight')  # Specify the filename and DPI

  # Compute the point biserial correlation coefficient
  corr_coef, p_value = pointbiserialr(comments['Aligned_Value'], comments['delta_awarded'])
  print(f'Correlation Coefficient: {corr_coef:.2f}')  
  print(f'P-Value: {p_value:.2f}')

if __name__ == '__main__':
  app()
