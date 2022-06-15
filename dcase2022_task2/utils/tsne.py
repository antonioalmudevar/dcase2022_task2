from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

def plot_tsne(x, y, n_classes, save_path):
    tsne = TSNE(n_components=2, learning_rate='auto', init='random')
    output = tsne.fit_transform(x)
    palette = sns.color_palette("hls", n_classes)
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x = output[:,0],
        y = output[:,1],
        hue=y,
        palette=palette,
        legend='full'
    )
    plt.savefig(save_path)
    plt.close()
    