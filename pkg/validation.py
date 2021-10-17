import matplotlib.pyplot as plt
def save_before_after(before, after, title, filename):
    fig, axs = plt.subplots(ncols=2, nrows=1)
    axs[0].imshow(before)
    axs[1].imshow(after)
    fig.suptitle(f'{title}')
    plt.savefig(f'{filename}')