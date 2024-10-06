import matplotlib.pyplot as plt
import umap
import umap.plot
import numpy as np

def umap_show(embedding_list, label_list, output_fp):
    embeddings = umap.UMAP(metric="cosine").fit(embedding_list)
    label_str_to_int_mapper = {label: i for i, label in enumerate(set(label_list))}
    label_arr = np.array([label_str_to_int_mapper[label] for label in label_list])
    
    plt.figure(figsize=(8, 6))
    umap.plot.points(
        embeddings,
        labels=label_arr,
        show_legend=False,
        color_key_cmap='Paired', background='black'
    )
    plt.savefig(str(output_fp))
    plt.close()