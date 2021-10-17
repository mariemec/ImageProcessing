from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

from pkg.validation import save_before_after
from pkg.zplane import zplane

FILENAME = "image_complete"
DESTINATION_PATH = "results/"


def read_npy_file(filename):
    img_load = np.load(filename)
    return img_load


def zp_to_ba(zeroes, poles):
    b = np.poly(zeroes)
    a = np.poly(poles)
    return b, a


def filter(data, b, a, file_extension):
    columns = data
    filtered_data = [signal.lfilter(b, a, col) for col in columns]
    plt.imsave(f'{DESTINATION_PATH}{FILENAME}_{file_extension}', filtered_data)
    return filtered_data


def rotate90clockwise(matrix):
    rows, columns = matrix.shape
    rotated_matrix = np.zeros(shape=(rows, columns))
    rotation_matrix = np.array([[0, 1], [-1, 0]])

    for (i, j), value in np.ndenumerate(matrix):
        rotated_vector_i, rotated_vector_j = np.matmul(rotation_matrix, np.array([i, j]))
        rotated_matrix[rotated_vector_i, rotated_vector_j] = matrix[i, j]

    plt.imsave(f'{DESTINATION_PATH}{FILENAME}_rotated.png', rotated_matrix)
    return rotated_matrix


def bilineaire():
    b = np.array([0.41816, 0.83633, 0.41816])
    a = np.array([1, 0.46294, 0.2097])
    return b, a, 2, "bilineaire"

def butterworth(wp, ws, gpass, gstop, f_e):
    N, Wn = signal.buttord(wp, ws, gpass, gstop, fs=f_e)
    [b, a] = signal.butter(N, Wn, fs=f_e)
    return b, a, N, "butterworth"


def cheb1(wp, ws, gpass, gstop, f_e):
    N, Wn = signal.cheb1ord(wp, ws, gpass, gstop, fs=f_e)
    [b, a] = signal.cheby1(N, gpass, Wn, fs=f_e)
    return b, a, N, "chebyshev1"


def cheb2(wp, ws, gpass, gstop, f_e):
    N, Wn = signal.cheb2ord(wp, ws, gpass, gstop, fs=f_e)
    [b, a] = signal.cheby2(N, gstop, Wn, fs=f_e)
    return b, a, N, "chebyshev2"


def elliptique(wp, ws, gpass, gstop, f_e):
    N, Wn = signal.ellipord(wp, ws, gpass, gstop, fs=f_e)
    [b, a] = signal.ellip(N, gpass, gstop, Wn, fs=f_e)
    return b, a, N, "elliptique"


def show_filters(img, b_a_N_filters, wp, ws, f_e, filename):
    num_of_filters = len(b_a_N_filters)
    fig, axs = plt.subplots(ncols=num_of_filters, nrows=2, figsize=(4*num_of_filters, 2*num_of_filters))

    wp_normalized = wp * 2 * np.pi / f_e
    ws_normalized = ws * 2 * np.pi / f_e

    for i, baN_type in enumerate(b_a_N_filters):
        b = baN_type[0]
        a = baN_type[1]
        N = baN_type[2]
        filter_type = baN_type[3]

        img_filtered = filter(img, b, a, f'{filter_type}.png')
        w, H_w = signal.freqz(b, a)

        axs[0, i].imshow(img_filtered)
        axs[1, i].plot(w, 20 * np.log10(abs(H_w)))
        axs[1, i].set_title(f'{filter_type} Ordre {N}')
        axs[1, i].axvline(wp_normalized, color='red', linestyle="--")
        axs[1, i].axvline(ws_normalized, color='orange', linestyle="--")
        axs[1, i].set_xlabel("fréquence normalisée (rad/ech)")

    for ax in axs.flat:
        ax.label_outer()

    axs[1, 0].set_ylabel("Amplitude [dB]")
    plt.tight_layout()
    plt.savefig(f'{filename}')


def compress(data, percentage):
    cov_matrix = np.cov(data)  # covariance matrix
    eigvalues, eigvectors = np.linalg.eig(cov_matrix)
    passage_matrix = eigvectors.T

    img_compressed = np.matmul(passage_matrix, data)
    compression_len = int(len(img_compressed) * percentage / 100)
    img_compressed[len(img_compressed) - compression_len:] = 0
    plt.imsave(f'{DESTINATION_PATH}{FILENAME}{percentage}_compressed.png', img_compressed)
    return img_compressed, passage_matrix


def decompress(data, passage_matrix, percentage):
    passage_matrix_inv = np.linalg.inv(passage_matrix)
    img_decompressed = np.matmul(passage_matrix_inv, data)
    plt.imsave(f'{DESTINATION_PATH}{FILENAME}{percentage}_decompressed.png', img_decompressed)

    return img_decompressed


if __name__ == '__main__':
    plt.gray()
    img_data = read_npy_file(f'media/{FILENAME}.npy')

    # Poles & Zeroes of H(z)
    z1 = 0.9 * np.e ** (1j * np.pi / 2)
    z2 = 0.9 * np.e ** (-1j * np.pi / 2)
    z3 = 0.95 * np.e ** (1j * np.pi / 8)
    z4 = 0.95 * np.e ** (-1j * np.pi / 8)
    p1 = 0
    p2 = -0.99
    p3 = -0.99
    p4 = 0.8

    # Get b, a for H(z)
    b_aberrations, a_aberrations = zp_to_ba([z1, z2, z3, z4], [p1, p2, p3, p4])

    # Filter aberrations (H(z)^-1) - Swap b and a
    img_without_aberrations = filter(img_data, a_aberrations, b_aberrations, "without_abberations.png")

    # Rotate 90deg clockwise
    img_rotated = rotate90clockwise(np.array(img_without_aberrations))

    # Filter high freq noise
    baN_bilin = bilineaire()     # Bilinear Method

    # Python Method
    wp = 500
    ws = 750
    gpass = 0.2
    gstop = 60
    f_e = 1600

    # Store (b, a, N, filter_type) in a tuple for each filter
    baN_butter = butterworth(wp, ws, gpass, gstop, f_e)
    baN_cheby1 = cheb1(wp, ws, gpass, gstop, f_e)
    baN_cheby2 = cheb2(wp, ws, gpass, gstop, f_e)
    baN_ellipt = butterworth(wp, ws, gpass, gstop, f_e)

    show_filters(img_rotated, [baN_bilin, baN_butter, baN_cheby1, baN_cheby2, baN_ellipt], wp, ws, f_e, filename=f'{DESTINATION_PATH}reponses-frequentielles.png')
    img_filtered_bilineaire = filter(img_rotated, baN_bilin[0], baN_bilin[1], f'{baN_bilin[3]}.png')
    img_filtered_elliptique = filter(img_rotated, baN_ellipt[0], baN_ellipt[1], f'{baN_ellipt[3]}.png')

    # Compression
    img_compressed_50, passage_matrix = compress(img_filtered_bilineaire, 50)
    img_compressed_75, passage_matrix = compress(img_filtered_bilineaire, 75)

    # Decompression
    img_decompressed_50 = decompress(img_compressed_50, passage_matrix, 50)
    img_decompressed_75 = decompress(img_compressed_75, passage_matrix, 75)

    validation = True
    if validation:
        # Aberration removal
        zplane(a_aberrations, b_aberrations, filename='validation/1-zplane-inverse.png')
        save_before_after(img_data, img_without_aberrations, 'Image avec et sans abbérations', 'validation/2-avec_sans_abberations.png')

        # Rotation
        save_before_after(img_without_aberrations, img_rotated, 'Image avant et après rotation', 'validation/3-avant_apres_rotation.png')

        # Bilinear transform vs Python - Should save automatically when calling try_all_filters
        zplane(baN_bilin[0], baN_bilin[1], "validation/4-zplane_bilineaire.png")
        zplane(baN_ellipt[0], baN_ellipt[1], "validation/6-zplane_elliptique.png")
        show_filters(img_rotated, [baN_bilin, baN_ellipt], wp, ws, f_e, "validation/578-reponses_frequentielles_bilineaire_elliptique.png")

        # Compression
        save_before_after(img_compressed_50, img_decompressed_50, 'Image compressée et décompressé (50%)', 'validation/9-compression50.png')
        save_before_after(img_compressed_75, img_decompressed_75, 'Image compressée et décompressé (75%)', 'validation/10-compression75.png')