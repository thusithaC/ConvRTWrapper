from conversation_model import ConvRTModel

from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def draw_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    plt.show()
    return f, ax, sc, txts


if __name__ == '__main__':
    model = ConvRTModel()
    sentences = ["hello my name this Tom", "hi there I'm Bert", "hello nice to meet you",
                 "Bye see you later", "catch you sometime later", "bye for now",
                 "Can you help me?", "I need assistance in this task", "is there someone who could help"]
    labels = np.array([0,0,0,1,1,1,2,2,2])

    encodings = model.get_embeddings(sentences)
    sentence_tsne = TSNE().fit_transform(encodings)
    draw_scatter(sentence_tsne, labels)