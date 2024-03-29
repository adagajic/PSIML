import numpy
import imageio
"""Some filtering functions from scipy.signal"""

import numpy as np
from typing import Tuple
from collections import deque as deques

# infinite print
np.set_printoptions(threshold=np.inf)
def buttap(N):
    """Return (z,p,k) for analog prototype of Nth-order Butterworth filter.

    The filter will have an angular (e.g., rad/s) cutoff frequency of 1.
    """
    if abs(int(N)) != N:
        raise ValueError("Filter order must be a nonnegative integer")
    z = np.array([])
    m = np.arange(-N+1, N, 2)
    # Middle value is 0 to ensure an exactly real pole
    p = -np.exp(1j * np.pi * m / (2 * N))
    k = 1
    return z, p, k

def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree

def lp2lp_zpk(z, p, k, wo=1.0):
    r"""
    Transform a lowpass filter prototype to a different frequency.

    Return an analog low-pass filter with cutoff frequency `wo`
    from an analog low-pass filter prototype with unity cutoff frequency,
    using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired cutoff, as angular frequency (e.g., rad/s).
        Defaults to no change.

    Returns
    -------
    z : ndarray
        Zeros of the transformed low-pass filter transfer function.
    p : ndarray
        Poles of the transformed low-pass filter transfer function.
    k : float
        System gain of the transformed low-pass filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)  # Avoid int wraparound

    degree = _relative_degree(z, p)

    # Scale all points radially from origin to shift cutoff frequency
    z_lp = wo * z
    p_lp = wo * p

    # Each shifted pole decreases gain by wo, each shifted zero increases it.
    # Cancel out the net change to keep overall gain the same
    k_lp = k * wo**degree

    return z_lp, p_lp, k_lp

def lp2bp_zpk(z, p, k, wo=1.0, bw=1.0):
    r"""
    Transform a lowpass filter prototype to a bandpass filter.

    Return an analog band-pass filter with center frequency `wo` and
    bandwidth `bw` from an analog low-pass filter prototype with unity
    cutoff frequency, using zeros, poles, and gain ('zpk') representation.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    wo : float
        Desired passband center, as angular frequency (e.g., rad/s).
        Defaults to no change.
    bw : float
        Desired passband width, as angular frequency (e.g., rad/s).
        Defaults to 1.

    Returns
    -------
    z : ndarray
        Zeros of the transformed band-pass filter transfer function.
    p : ndarray
        Poles of the transformed band-pass filter transfer function.
    k : float
        System gain of the transformed band-pass filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    wo = float(wo)
    bw = float(bw)

    degree = _relative_degree(z, p)

    # Scale poles and zeros to desired bandwidth
    z_lp = z * bw/2
    p_lp = p * bw/2

    # Square root needs to produce complex result, not NaN
    z_lp = z_lp.astype(complex)
    p_lp = p_lp.astype(complex)

    # Duplicate poles and zeros and shift from baseband to +wo and -wo
    z_bp = np.concatenate((z_lp + np.sqrt(z_lp**2 - wo**2),
                           z_lp - np.sqrt(z_lp**2 - wo**2)))
    p_bp = np.concatenate((p_lp + np.sqrt(p_lp**2 - wo**2),
                           p_lp - np.sqrt(p_lp**2 - wo**2)))

    # Move degree zeros to origin, leaving degree zeros at infinity for BPF
    z_bp = np.append(z_bp, np.zeros(degree))

    # Cancel out gain change from frequency scaling
    k_bp = k * bw**degree

    return z_bp, p_bp, k_bp

def bilinear_zpk(z, p, k, fs):
    r"""
    Return a digital IIR filter from an analog one using a bilinear transform.

    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``2*fs*(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.

    Parameters
    ----------
    z : array_like
        Zeros of the analog filter transfer function.
    p : array_like
        Poles of the analog filter transfer function.
    k : float
        System gain of the analog filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g., hertz). No prewarping is
        done in this function.

    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)
    fs = float(fs)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z

def zpk2tf(z, p, k):
    r"""
    Return polynomial transfer function representation from zeros and poles

    Parameters
    ----------
    z : array_like
        Zeros of the transfer function.
    p : array_like
        Poles of the transfer function.
    k : float
        System gain.

    Returns
    -------
    b : ndarray
        Numerator polynomial coefficients.
    a : ndarray
        Denominator polynomial coefficients.
    """
    z = np.atleast_1d(z)
    k = np.atleast_1d(k)
    if len(z.shape) > 1:
        temp = np.poly(z[0])
        b = np.empty((z.shape[0], z.shape[1] + 1), temp.dtype.char)
        if len(k) == 1:
            k = [k[0]] * z.shape[0]
        for i in range(z.shape[0]):
            b[i] = k[i] * np.poly(z[i])
    else:
        b = k * np.poly(z)
    a = np.atleast_1d(np.poly(p))

    # Use real output if possible. Copied from np.poly, since
    # we can't depend on a specific version of np.
    if issubclass(b.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(z, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) == np.sort_complex(pos_roots)):
                b = b.real.copy()

    if issubclass(a.dtype.type, np.complexfloating):
        # if complex roots are all complex conjugates, the roots are real.
        roots = np.asarray(p, complex)
        pos_roots = np.compress(roots.imag > 0, roots)
        neg_roots = np.conjugate(np.compress(roots.imag < 0, roots))
        if len(pos_roots) == len(neg_roots):
            if np.all(np.sort_complex(neg_roots) ==
                         np.sort_complex(pos_roots)):
                a = a.real.copy()

    return b, a

def butterworth_lowpass(N: int, Wn: float) -> Tuple[np.ndarray, np.ndarray]:
    """LP Butterworth digital filter

    Args:
        N (int): order
        Wn (float): cutoff frequency

    Returns:
        b : ndarray
            Numerator polynomial coefficients.
        a : ndarray
            Denominator polynomial coefficients.
    """
    assert np.size(Wn) == 1, "Must specify a single critical frequency Wn for lowpass or highpass filter"
    assert np.all(Wn > 0) and np.all(Wn < 1), "Digital filter critical frequencies must be 0 < Wn < 1"
    
    z, p, k = buttap(N)
    warped = 4 * np.tan(np.pi * Wn / 2) # digital
    z, p, k = lp2lp_zpk(z, p, k, wo=warped)
    z, p, k = bilinear_zpk(z, p, k, fs=2)
    b, a = zpk2tf(z, p, k)

    return b, a

def butterworth_bandpass(N: int, Wn: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """LP Butterworth digital filter

    Args:
        N (int): order
        Wn (Tuple[float, float]): band pass frequencies

    Returns:
        b : ndarray
            Numerator polynomial coefficients.
        a : ndarray
            Denominator polynomial coefficients.
    """
    Wn = np.array(Wn)
    assert np.size(Wn) == 2, "Must specify a single critical frequency Wn for lowpass or highpass filter"
    assert np.all(Wn > 0) and np.all(Wn < 1), "Digital filter critical frequencies must be 0 < Wn < 1"
    
    z, p, k = buttap(N)
    warped = 4 * np.tan(np.pi * Wn / 2) # digital
    
    bw = warped[1] - warped[0]
    wo = np.sqrt(warped[0] * warped[1])
    z, p, k = lp2bp_zpk(z, p, k, wo=wo, bw=bw)
    z, p, k = bilinear_zpk(z, p, k, fs=2)
    b, a = zpk2tf(z, p, k)

    return b, a


def lfilter(b, a, x):
    """A simple implementation of a 1-D linear filter."""
    a = np.array(a)
    b = np.array(b)
    y = np.zeros_like(x)
    a0 = a[0]
    if a0 != 1:
        a = a / a0
        b = b / a0
    for i in range(len(x)):
        for j in range(len(b)):
            if i - j >= 0:
                y[i] += b[j] * x[i - j]
        for j in range(1, len(a)):
            if i - j >= 0:
                y[i] -= a[j] * y[i - j]
    return y

def bfs_white_pixel(img, first_pixel):
    queue = deques()
    visited = set()
    queue.append(first_pixel)
    visited.add(first_pixel)
    while queue:
        pixel = queue.popleft()
        if img[pixel[0]][pixel[1]] == 0:
            continue
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pixel = (pixel[0] + dx, pixel[1] + dy)
            if (0 <= new_pixel[0] < len(img) and 0 <= new_pixel[1] < len(img[0]) and
                    new_pixel not in visited and img[new_pixel[0]][new_pixel[1]] == 255):
                visited.add(new_pixel)
                queue.append(new_pixel)
    return visited

def get_num_frames(video):
    i = 0
    try:
        while True:
            video.get_data(i)
            i += 1
    except IndexError:
        return i
    
def all_peaks(y):
    # use power of 2 for fft to make it faster
    # transfer linear scale to power scale
    pow = np.sum(np.abs(y) ** 2) / len(y)
    interval = 50
    new_y = np.zeros(len(y) - interval)
    for i in range(len(y) - interval):
        new_y[i] = np.sum(y[i:i + interval])
    peaks = []
    for i in range(1, len(new_y) - 1):
        if (new_y[i] > 100 * pow) and new_y[i] > new_y[i - 1] and new_y[i] > new_y[i + 1]:
            # check for 600 neighbours
            is_peak = True
            for j in range(1, 600):
                if i - j >= 0 and new_y[i - j] > new_y[i]:
                    is_peak = False
                    break
                if i + j < len(new_y) and new_y[i + j] > new_y[i]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
    return peaks  
                
        
    return peaks
if __name__ == '__main__':
    # read the video from mp4 file
    video_dir = input()
    audio_dir = input()
    video = imageio.get_reader(video_dir)
    # get audo from .npy file
    audio = np.load(audio_dir)
    # get number of frames
    num_frames = get_num_frames(video)
    step = 0
    gray = None
    gray_prev = None
    first_frame = True
    result = np.ones((video.get_data(0).shape[0], video.get_data(0).shape[1]), dtype=bool)

    for i in range(num_frames):
        frame = video.get_data(i)
        gray = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140])
        gray = (gray * 255 / np.max(gray)).astype('uint8')

        # Convert to black and white
        gray[gray > 60] = 255
        gray[gray != 255] = 0

        # Perform an element-wise logical AND operation with the result array
        result = result & (gray == 255)
    gray = (result * 255).astype('uint8')
    # get average locatopn of white pixels
    all_white_pixels = np.argwhere(gray == 255)
    # get all from all_white_pixels_by_1 that are not in all_white_pixels
    all_white_pixels = set(map(tuple, all_white_pixels))
    to_be_removed = set()
    for pixel in all_white_pixels:
        count_white = 0
        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_pixel = (pixel[0] + dx, pixel[1] + dy)
            if new_pixel in all_white_pixels:
                count_white += 1
        if count_white < 3:
            print(pixel)
            to_be_removed.add(pixel)
    for pixel in to_be_removed:
        all_white_pixels.remove(pixel)
    first_frame = video.get_data(0)
    first_img = np.dot(first_frame[...,:3], [0.2989, 0.5870, 0.1140])
    first_img = (first_img * 255 / np.max(first_img)).astype('uint8')
    first_img[(first_img > 60)] = 255
    first_img[first_img != 255] = 0

    first_white_frame_circle = None
    finished = False
    for j in range(first_img.shape[1]):
        for i in range(first_img.shape[0]):
            if first_img[i][j] == 255 and (i, j) not in all_white_pixels:
                first_white_frame_circle = (i, j)
                finished = True
                break
        if finished:
            break
    visited = bfs_white_pixel(first_img, first_white_frame_circle)
    only_circle_img = np.zeros_like(first_img)
    for pixel in visited:
        only_circle_img[pixel[0]][pixel[1]] = 255
    x_circ = 0
    y_circ = 0
    count_circ = 0
    for pixel in visited:
        x_circ += pixel[0]
        y_circ += pixel[1]
        count_circ += 1
    x_circ = x_circ // count_circ
    y_circ = y_circ // count_circ
    print(x_circ, y_circ)
    
    x = 0
    y = 0
    count = 0
    for pixel in all_white_pixels:
        x += pixel[0]
        y += pixel[1]
        count += 1
    x = x // count
    y = y // count
    print(x, y)
    
    special_bounce = set()
    
    max_x_diff = 0
    max_y_diff = 0
    for i in range(gray.shape[0]):
        if (i, y) in all_white_pixels:
            x_diff = abs(i - x)
            max_x_diff = max(max_x_diff, x_diff)
    for i in range(gray.shape[1]):
        if (x, i) in all_white_pixels:
            y_diff = abs(i - y)
            max_y_diff = max(max_y_diff, y_diff)
    max_x_diff += 3
    max_y_diff += 3
    x_shape = 2 * (max_x_diff - x) + 1
    y_shape = 2 * (max_y_diff - y) + 1
    for i in range(x - max_x_diff, x + max_x_diff + 1):
        special_bounce.add((i, y - max_y_diff))
        special_bounce.add((i, y + max_y_diff))
    for i in range(y - max_y_diff, y + max_y_diff + 1):
        special_bounce.add((x - max_x_diff, i))
        special_bounce.add((x + max_x_diff, i))
    for i in range(gray.shape[0]):
            special_bounce.add((i, 2))
            special_bounce.add((i, gray.shape[1] - 3))
    for i in range(gray.shape[1]):
            special_bounce.add((2, i))
            special_bounce.add((gray.shape[0] - 3, i))
            
    special_bounce_img = np.zeros_like(gray)
    for pixel in special_bounce:
        special_bounce_img[pixel[0]][pixel[1]] = 255
    
    
    # get all bounces in video, bounce happens when the white circle hits the special bounce
    # if bounce happens, disable special bounce pixels +-5 pixels for 5 frames
    disabled = {} # key is tuple of (i, j) and value is disabled_for
    disabled_for = 20
    disabled_around = 40
    count = 0
    bounces = 0
    bounced_pixels = set()
    while count < num_frames:
        frame = video.get_data(count)
        gray = np.dot(frame[...,:3], [0.2989, 0.5870, 0.1140])
        gray = (gray * 255 / np.max(gray)).astype('uint8')
        gray[(gray > 60)] = 255
        gray[gray != 255] = 0
        # decrement disabled_for for all disabled pixels
        for key in list(disabled.keys()):
            disabled[key] -= 1
            if disabled[key] == 0:
                del disabled[key]
        list_disabled = list(disabled.keys())
        
        # check if the white circle hits the special bounce
        # if it does, disable +-5 pixels for 5 frames
        
        for (i, j) in special_bounce:
            if gray[i][j] == 255:
                if (i, j) in disabled:
                    continue
                for dx in range(-disabled_around, disabled_around + 1):
                    for dy in range(-disabled_around, disabled_around + 1):
                        if i + dx >= 0 and i + dx < gray.shape[0] and j + dy >= 0 and j + dy < gray.shape[1] and (i + dx, j + dy) in special_bounce:
                            disabled[(i + dx, j + dy)] = disabled_for
                bounces += 1
                bounced_pixels.add((i, j))
        count += 1
    Fsaudio = 44100
    t = np.arange(0, len(audio)/Fsaudio, 1/Fsaudio) 
    
    low = 200
    high = 2000
    
    b, a = butterworth_bandpass(4, (low/(Fsaudio/2), high/(Fsaudio/2)))
    y = lfilter(b, a, audio)
    print(len(all_peaks(y)))
    print(bounces)
    all_bounces_img = np.zeros_like(gray)
    for pixel in bounced_pixels:
        all_bounces_img[pixel[0]][pixel[1]] = 255
    # image save all special bounces]
    special_bounces_img = np.zeros_like(gray)
    for pixel in special_bounce:
        special_bounces_img[pixel[0]][pixel[1]] = 255    
    