import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from pathlib import Path


def main():
    results_path = Path('results')

    plot_token_counts(results_path)
    plt.savefig(results_path / 'tokens.png')

    plot_dates(results_path)
    plt.savefig(results_path / 'dates.png')

    plot_languages(results_path)
    plt.savefig(results_path / 'language_detection.png')

    plot_domains(results_path)
    plt.savefig(results_path / 'domains.png')


def plot_token_counts(results_path):
    df = pd.read_csv(results_path / 'tokens.tsv',
                     sep='\t',
                     header=None,
                     names=['Number of tokens', 'Count'])

    print('Number of tokens:')
    print(f'Min: {df["Number of tokens"].min()}')
    print(f'Max: {df["Number of tokens"].max()}')
    print(f'Mean: {(df["Count"] * df["Number of tokens"]).sum()/df["Count"].sum():.1f}  ')

    plt.figure()
    sns.histplot(df, x='Number of tokens', weights='Count', bins=25, binrange=[0, 4000])


def plot_dates(results_path):
    df = pd.read_csv(results_path / 'dates.tsv',
                     sep='\t',
                     header=None,
                     parse_dates=[0],
                     names=['Date', 'Count'])

    plt.figure()
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

    plt.figure()
    sns.barplot(df, x="Detected language", y="Count")


def plot_domains(results_path):
    df = pd.read_csv(results_path / 'domains_annotated.tsv',
                     sep='\t',
                     header=None,
                     names=['Domain', 'Count', 'Content class'])
    df = df.sort_values(by='Count', ascending=False)

    # cumulative domain size
    plt.figure()
    cumsum = df['Count'].cumsum()
    cumsum = pd.concat([pd.Series([0]), cumsum])

    ax = sns.lineplot(x=np.arange(len(cumsum)), y=cumsum, drawstyle='steps-pre')
    ax.set_title('Cumulative document count per domain')
    ax.set_ylabel('Documents')
    ax.set_xlabel('Domain rank')

    # Treemap of largest domains
    plt.figure()
    df2 = df.dropna()
    df3 = df2[['Content class', 'Count']].groupby('Content class').sum().reset_index()
    prop_pages = df2['Count'].sum()/df['Count'].sum()

    palette = sns.color_palette('muted')
    ax = squarify.plot(sizes=df3['Count'], label=df3['Content class'], color=palette)
    ax.set_title(f'{len(df2)} largest domains ({100*prop_pages:.0f}% of pages)')
    plt.axis('off')


if __name__ == '__main__':
    main()
