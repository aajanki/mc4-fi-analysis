import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import squarify
from pathlib import Path


def main():
    results_path = Path('results')
    plot_token_counts(results_path)
    plot_dates(results_path)
    plot_languages(results_path)
    plot_topics(results_path)
    plot_top_domain_page_counts(results_path)


def plot_token_counts(results_path):
    df = pd.read_csv(results_path / 'tokens.tsv',
                     sep='\t',
                     header=None,
                     names=['Number of tokens', 'Count'])

    print(f'Number of documents: {df["Count"].sum()}')
    print(
        'Number of tokens per document: '
        f'min {df["Number of tokens"].min()}, '
        f'max {df["Number of tokens"].max()}, '
        f'mean {(df["Count"] * df["Number of tokens"]).sum() / df["Count"].sum():.1f}'
    )

    plt.figure()
    ax = sns.histplot(df, x='Number of tokens', weights='Count', bins=25, binrange=[0, 4000])
    ax.set_ylabel('Number of documents')

    plt.savefig(results_path / 'tokens.png')


def plot_dates(results_path):
    df = pd.read_csv(results_path / 'dates.tsv',
                     sep='\t',
                     header=None,
                     parse_dates=[0],
                     names=['Date', 'Count'])

    plt.figure()
    ax = sns.histplot(df, x='Date', weights='Count', bins=25)
    ax.set_xlabel('')
    ax.set_ylabel('Number of documents')

    plt.savefig(results_path / 'dates.png')


def plot_languages(results_path):
    df = pd.read_csv(results_path / 'language_detection.tsv',
                     sep='\t',
                     header=None,
                     names=['Detected language', 'Count'])

    fi_proportion = (df[df["Detected language"] == "fi-fi"]["Count"] / df["Count"].sum()).values[0]

    print(f'Proportion of documents very likely in Finnish: {100 * fi_proportion:.1f} %')

    plt.figure()
    ax = sns.barplot(df, x="Detected language", y="Count")
    ax.set_ylabel('Number of documents')

    plt.savefig(results_path / 'language_detection.png')


def plot_topics(results_path):
    df = pd.read_csv(results_path / 'domains_annotated.tsv',
                     sep='\t',
                     header=None,
                     low_memory=False,
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

    plt.savefig(results_path / 'domains_cumulative.png')

    # Treemap of largest domains
    plt.figure()
    df2 = df.dropna()
    df3 = df2[['Content class', 'Count']].groupby('Content class').sum().reset_index()
    prop_pages = df2['Count'].sum()/df['Count'].sum()

    palette = sns.color_palette('muted')
    ax = squarify.plot(sizes=df3['Count'], label=df3['Content class'], color=palette)
    ax.set_title(f'{len(df2)} suurinta verkkotunnusta ({100*prop_pages:.0f}% kaikista sivuista)')
    plt.axis('off')

    plt.savefig(results_path / 'topics_treemap.png')


def plot_top_domain_page_counts(results_path):
    df = pd.read_csv(results_path / 'domains.tsv',
                     sep='\t',
                     header=None,
                     low_memory=False,
                     names=['Domain', 'Count'])
    df_top = df.loc[:25].copy()
    df_top['Count'] = df_top['Count'] / 1_000_000

    plt.figure()
    ax = sns.barplot(df_top, x='Count', y='Domain', color='b')
    ax.spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    ax.set_ylabel('')
    ax.set_xlabel('Verkkosivujen lukumäärä (miljoonaa)')

    plt.savefig(results_path / 'top_domain_sizes.png')


if __name__ == '__main__':
    main()
