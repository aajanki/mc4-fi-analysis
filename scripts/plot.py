import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    results_path = Path('results')

    plt.figure()
    plot_token_counts(results_path)
    plt.savefig(results_path / 'tokens.png')

    plt.figure()
    plot_dates(results_path)
    plt.savefig(results_path / 'dates.png')

    plt.figure()
    plot_languages(results_path)
    plt.savefig(results_path / 'language_detection.png')


def plot_token_counts(results_path):
    df = pd.read_csv(results_path / 'tokens.tsv',
                     sep='\t',
                     header=None,
                     names=['Number of tokens', 'Count'])

    print('Number of tokens:')
    print(f'Min: {df["Number of tokens"].min()}')
    print(f'Max: {df["Number of tokens"].max()}')
    print(f'Mean: {(df["Count"] * df["Number of tokens"]).sum()/df["Count"].sum():.1f}  ')

    sns.histplot(df, x='Number of tokens', weights='Count', bins=25, binrange=[0, 4000])


def plot_dates(results_path):
    df = pd.read_csv(results_path / 'dates.tsv',
                     sep='\t',
                     header=None,
                     parse_dates=[0],
                     names=['Date', 'Count'])

    ax = sns.histplot(df, x='Date', weights='Count', bins=25)
    ax.set_title('dates')
    ax.set_xlabel('')


def plot_languages(results_path):
    df = pd.read_csv(results_path / 'language_detection.tsv',
                     sep='\t',
                     header=None,
                     names=['Detected language', 'Count'])

    fi_proportion = (df[df["Detected language"] == "fi-fi"]["Count"] / df["Count"].sum()).values[0]

    print(f'Likely Finnish: {100 * fi_proportion:.1f} %')

    sns.barplot(df, x="Detected language", y="Count")


if __name__ == '__main__':
    main()
