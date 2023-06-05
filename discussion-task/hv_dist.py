import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import typer
from scipy.stats import chi2_contingency

app = typer.Typer(add_completion=False)

def cramers_v(contingency_table):
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = np.sum(contingency_table)
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corrected = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corrected = r - ((r - 1) ** 2) / (n - 1)
    k_corrected = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2_corrected / min((k_corrected - 1), (r_corrected - 1)))

@app.command()
def main(
        results_file: str = typer.Option(None,help='path for inference file')
):

    pd.set_option('display.max_columns', None)

    data = pd.read_csv(results_file)
    data = data[(data["delta_thread"]=="True") | (data["delta_thread"]==True)]
    data['delta_thread'] = data['delta_thread'].astype(bool)
    data['delta_awarded'] = data['delta_awarded'].map({'True': True, 'False': False, True:True, False:False})
    data['delta_awarded'] = data['delta_awarded'].astype(bool)

    #Only comments 
    data = data[~(data["parent_id"].isnull())]

    columns_subset = ['Benevolence: dependability',
           'Achievement', 'Benevolence: caring', 'Hedonism',
           'Universalism: objectivity', 'Humility', 'Security: societal',
           'Conformity: interpersonal', 'Self-direction: thought',
           'Power: resources', 'Face', 'Power: dominance', 'Universalism: nature',
           'Universalism: tolerance', 'Stimulation', 'Security: personal',
           'Tradition', 'Universalism: concern', 'Conformity: rules',
           'Self-direction: action']

    data['Human Value'] = data[columns_subset].idxmax(axis=1)
    data['Top3'] = data[columns_subset].apply(lambda x: x.nlargest(3).index.tolist(), axis=1)

    # Subset with delta
    subset1 = data[data['delta_awarded'] == True]
    HV_list_sub1 = pd.Series([item for sublist in subset1['Top3'] for item in sublist])

    # Subset without delta
    subset2 = data[data['delta_awarded'] == False]
    HV_list_sub2 = pd.Series([item for sublist in subset2['Top3'] for item in sublist])

    # Calculate the percentage distribution of HVs for each subset
    subset1_counts = HV_list_sub1.value_counts(normalize=True) * 100
    subset2_counts = HV_list_sub2.value_counts(normalize=True) * 100


    # Get the unique HVs from both subsets
    categories = np.union1d(subset1_counts.index, subset2_counts.index)

    # Fill missing with zeros
    subset1_counts = subset1_counts.reindex(categories, fill_value=0)
    subset2_counts = subset2_counts.reindex(categories, fill_value=0)

    # Set the width of the bars
    bar_width = 0.35

    # Set the x-axis positions for the bars
    x = np.arange(len(categories))

    # Plot the comparison of category distributions
    plt.figure(figsize=(16, 5))
    plt.bar(x - bar_width/2, subset1_counts, width=bar_width, label='Delta')
    plt.bar(x + bar_width/2, subset2_counts, width=bar_width, label='Non-Delta')
    plt.title('Comparison of Human Value Distributions in Delta vs Non-delta Comments')
    plt.xlabel('Human Value')
    plt.ylabel('Percentage')
    plt.xticks(x, categories, rotation=45,ha='right')
    plt.legend(title='Subset')
    #plt.show()

    plt.savefig('results/human_value_dist.png', dpi=300, bbox_inches='tight')  # Specify the desired filename and DPI
    
    data = data[['delta_awarded','Top3']]
    data = data.explode('Top3')

    contingency_table = pd.crosstab(data['delta_awarded'], data['Top3'])

    cramers_v_value = cramers_v(contingency_table.values)
    print("Cramér's V:", cramers_v_value)

if __name__ == '__main__':
    app()
