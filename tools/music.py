"""Original soundtrack for the Casio Arabic Dial spot.

Everything is synthesised from scratch (no samples): an upbeat Arabic-pop /
electro track in maqam Hijaz on D, 120 BPM, with darbuka maqsum groove,
plucked qanun/oud (Karplus-Strong), a ney/mizmar-style lead, bass, pads and
sound-design hits synced to the picture edit (see ad/index.html, same grid).

Output: build/music.wav (48 kHz, stereo, 16-bit)
"""
import numpy as np
from numba import njit
from scipy.signal import butter, sosfilt, fftconvolve
import wave, os

SR = 48000
BPM = 120
BEAT = 60 / BPM          # 0.5 s
BAR = 4 * BEAT           # 2 s
S16 = BEAT / 4           # 16th note
DUR = 33.5
N = int(DUR * SR)
rng = np.random.default_rng(7)

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)

# ----------------------------------------------------------------- buses
buses = {k: np.zeros((2, N)) for k in ('drums', 'bass', 'synth', 'fx')}
send = np.zeros((2, N))      # reverb send
dsend = np.zeros((2, N))     # delay send

def add(bus, sig, t, gain=1.0, pan=0.0, rev=0.0, dly=0.0):
    """Place a mono or stereo signal at time t (s) on a bus."""
    i0 = int(round(t * SR))
    if i0 >= N:
        return
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)
    n = min(sig.shape[1], N - i0)
    if i0 < 0:
        sig = sig[:, -i0:]; n = min(sig.shape[1], N); i0 = 0
    s = sig[:, :n] * gain
    buses[bus][:, i0:i0 + n] += s
    if rev:
        send[:, i0:i0 + n] += s * rev
    if dly:
        dsend[:, i0:i0 + n] += s * dly

def tt(n):
    return np.arange(int(n * SR)) / SR

def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)

def hp(x, f, order=2):
    return sosfilt(butter(order, f, 'high', fs=SR, output='sos'), x)

def lp(x, f, order=2):
    return sosfilt(butter(order, f, 'low', fs=SR, output='sos'), x)

def noise(d):
    return rng.uniform(-1, 1, int(d * SR))

# ----------------------------------------------------------- DSP kernels
@njit(cache=True)
def svf_lp(x, cut, res):
    """Chamberlin state-variable low-pass, 2x oversampled, per-sample cutoff."""
    y = np.zeros_like(x); low = 0.0; band = 0.0
    q = max(2.0 * (1.0 - res), 0.1)
    for i in range(x.shape[0]):
        f = 2.0 * np.sin(np.pi * min(cut[i], 16000.0) / 96000.0)
        for _ in range(2):
            low += f * band
            high = x[i] - low - q * band
            band += f * high
        y[i] = low
    return y

@njit(cache=True)
def ks_pluck(freq, n, bright, decay, seed):
    """Karplus-Strong string with allpass fractional delay (accurate tuning)."""
    np.random.seed(seed)
    P = 48000.0 / freq - 0.5          # averager adds half a sample
    D = int(np.floor(P)); frac = P - D
    if frac < 0.15:
        D -= 1; frac += 1.0
    c = (1 - frac) / (1 + frac)
    buf = np.random.uniform(-1, 1, D)
    for k in range(1, D):             # soften excitation
        buf[k] = bright * buf[k] + (1 - bright) * buf[k - 1]
    buf -= buf.mean()
    out = np.zeros(n)
    idx = 0; prev = 0.0; x1 = 0.0; y1 = 0.0
    for i in range(n):
        cur = buf[idx]
        out[i] = cur
        avg = decay * 0.5 * (cur + prev)
        prev = cur
        ap = c * avg + x1 - c * y1
        x1 = avg; y1 = ap
        buf[idx] = ap
        idx += 1
        if idx >= D:
            idx = 0
    return out

@njit(cache=True)
def osc_saw_blep(freq_arr):
    """Band-limited sawtooth (polyBLEP) with per-sample frequency."""
    n = freq_arr.shape[0]; y = np.zeros(n); ph = 0.0
    for i in range(n):
        dt = freq_arr[i] / 48000.0
        ph += dt
        if ph >= 1.0:
            ph -= 1.0
        v = 2.0 * ph - 1.0
        if ph < dt:
            t = ph / dt; v -= t + t - t * t - 1.0
        elif ph > 1.0 - dt:
            t = (ph - 1.0) / dt; v -= t * t + t + t + 1.0
        y[i] = v
    return y

def env_adsr(n, a, d, s, r, hold=None):
    t = np.arange(n) / SR
    hold = (n / SR - r) if hold is None else hold
    e = np.where(t < a, t / max(a, 1e-4), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    rel = np.clip((t - hold) / max(r, 1e-4), 0, 1)
    return e * (1 - rel) ** 2

# ------------------------------------------------------------ instruments
def kick(g=1.0):
    t = tt(0.55)
    f = 48 + 110 * np.exp(-t / 0.035)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t / 0.2)
    click = hp(noise(0.55), 2500) * np.exp(-t / 0.004) * 0.35
    return np.tanh(1.6 * (body + click)) * g

def doum():
    t = tt(0.5)
    f = 88 + 60 * np.exp(-t / 0.02)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.22)
    skin = bp(noise(0.5), 150, 700) * np.exp(-t / 0.03) * 0.5
    return np.tanh(1.3 * (body + skin))

def tek(soft=1.0):
    t = tt(0.18)
    ring = np.sin(2 * np.pi * 820 * t) * np.exp(-t / 0.018) * 0.6
    ring += np.sin(2 * np.pi * 1930 * t) * np.exp(-t / 0.012) * 0.3
    snap = bp(noise(0.18), 2500, 9000) * np.exp(-t / 0.012)
    return (ring + snap) * soft

def clap():
    t = tt(0.4); x = np.zeros_like(t)
    nz = bp(noise(0.4), 900, 3200, 2)
    for k, dt in enumerate([0, 0.011, 0.022]):
        i = int(dt * SR)
        x[i:] += nz[: len(x) - i] * np.exp(-(t[: len(x) - i]) / 0.006)
    x += nz * np.exp(-np.maximum(t - 0.03, 0) / 0.11) * (t > 0.03) * 0.7
    return x

def hat(open_=False):
    d = 0.35 if open_ else 0.06
    t = tt(d)
    m = sum(np.sign(np.sin(2 * np.pi * f * t)) for f in (3140, 4532, 5810, 7340, 8720)) / 5
    x = hp(m * 0.5 + noise(d) * 0.7, 7000, 3)
    return x * np.exp(-t / (0.09 if open_ else 0.014))

def riq():
    """Tambourine-like jingles."""
    t = tt(0.16)
    m = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) for f in (5200, 6320, 7900, 9100, 10450))
    x = hp(m * 0.2 + noise(0.16) * 0.4, 5000, 2)
    return x * (1 - np.exp(-t / 0.002)) * np.exp(-t / 0.05)

def tick(pitch=1.0, g=1.0):
    t = tt(0.08)
    x = np.sin(2 * np.pi * 3600 * pitch * t) * np.exp(-t / 0.004)
    x += np.sin(2 * np.pi * 1450 * pitch * t) * np.exp(-t / 0.009) * 0.6
    x += hp(noise(0.08), 4000) * np.exp(-t / 0.0015) * 0.8
    return x * g

def crash():
    d = 2.6; t = tt(d)
    m = sum(np.sign(np.sin(2 * np.pi * f * t + rng.uniform(0, 6))) for f in (3300, 4710, 5960, 7210, 8840, 10100))
    x = hp(m * 0.25 + noise(d), 3500, 2)
    return x * np.exp(-t / 0.75) * (1 - np.exp(-t / 0.002))

def sub_boom(d=2.0, f0=55):
    t = tt(d)
    f = f0 + 40 * np.exp(-t / 0.08)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / (d / 3.5))

def riser(d, f0=300, f1=6000):
    t = tt(d); n = len(t)
    x = noise(d)
    cut = f0 * (f1 / f0) ** (t / d)
    y = svf_lp(x, cut, 0.55) - svf_lp(x, cut * 0.35, 0.0)
    tone = np.sin(2 * np.pi * np.cumsum(180 * 2 ** (2 * t / d)) / SR) * 0.15
    e = (t / d) ** 2.2
    return (y + tone) * e

def whoosh(d=0.45, peak=0.75, f0=500, f1=5000):
    t = tt(d); x = noise(d)
    pos = np.clip(t / d, 0, 1)
    cut = f0 + (f1 - f0) * np.sin(np.pi * pos) ** 2
    y = svf_lp(x, cut, 0.4) - lp(x, f0 * 0.6)
    e = np.where(pos < peak, (pos / peak) ** 2, ((1 - pos) / (1 - peak)) ** 1.5)
    return y * e

def rev_cymbal(d=1.0):
    c = crash()[: int(d * SR)][::-1]
    return c * np.linspace(0, 1, len(c)) ** 1.5

def shimmer(d=3.0, f0=1318.5):
    t = tt(d); x = np.zeros_like(t)
    for r, a, dec in [(1, 1, 1.2), (2.76, 0.5, 0.7), (5.40, 0.3, 0.4), (8.93, 0.18, 0.25), (2.0, 0.4, 1.0), (3.01, 0.2, 0.6)]:
        x += a * np.sin(2 * np.pi * f0 * r * t + r) * np.exp(-t / dec)
    return x * (1 - np.exp(-t / 0.001)) * 0.35

def pluck(m, d=1.6, bright=0.75, decay=0.996, seed=1):
    f = mtof(m)
    x = ks_pluck(f, int(d * SR), bright, decay, seed)
    x = x * env_adsr(len(x), 0.001, 1, 1, 0.08)
    return hp(x, 120)

def qanun(m, d=1.2, seed=1):
    a = pluck(m, d, 0.85, 0.997, seed)
    b = pluck(m + 12, d, 0.7, 0.995, seed + 7) * 0.35
    return (a + b) * 0.8

def oud(m, d=2.2, seed=3):
    x = pluck(m, d, 0.45, 0.998, seed)
    body = bp(x, 180, 900)
    return x * 0.7 + body * 0.8

def bass_note(m, d):
    n = int(d * SR); t = np.arange(n) / SR
    f = np.full(n, mtof(m))
    saw = osc_saw_blep(f)
    cut = 180 + 1400 * np.exp(-t / 0.07)
    y = svf_lp(saw, cut, 0.35) * 0.8
    sub = np.sin(2 * np.pi * mtof(m) * t) * 0.85
    e = env_adsr(n, 0.003, 0.2, 0.75, 0.03)
    return np.tanh(1.4 * (y + sub) * e) * 0.9

def lead_phrase(notes, t0_steps):
    """Ney / mizmar-ish lead with portamento, delayed vibrato and breath."""
    end = max(s + l for s, _, l in notes)
    n = int((end * S16 + 0.6) * SR)
    t = np.arange(n) / SR
    f = np.zeros(n); amp = np.zeros(n); vib_t = np.zeros(n)
    for k, (s, m, l) in enumerate(notes):
        i0 = int(s * S16 * SR); i1 = int((s + l) * S16 * SR)
        target = mtof(m)
        prev = mtof(notes[k - 1][1]) if k else target
        g = np.arange(i1 - i0) / SR
        glide = np.exp(-g / 0.035)
        f[i0:i1] = target * (prev / target) ** glide
        vib_t[i0:i1] = g
        a = np.minimum(g / 0.03, 1) * np.where(g > (l * S16 - 0.05), np.maximum(0, 1 - (g - (l * S16 - 0.05)) / 0.05), 1)
        amp[i0:i1] = np.maximum(amp[i0:i1], a)
    f[int(end * S16 * SR):] = f[int(end * S16 * SR) - 1]
    vib = 1 + 0.012 * np.minimum(vib_t / 0.35, 1) * np.sin(2 * np.pi * 5.6 * t)
    f = f * vib
    saw = osc_saw_blep(f)
    sq = np.sign(np.sin(2 * np.pi * np.cumsum(f * 1.0) / SR)) * 0.3
    x = svf_lp(saw * 0.7 + sq, np.full(n, 3200.0), 0.5)
    x = x + bp(x, 1000, 1800) * 1.2           # nasal formant
    breath = bp(noise(n / SR), 1500, 6000) * 0.05
    amp = lp(amp, 40)
    return (x + breath) * amp * 0.5

def pad_chord(ms, d, bright=1600):
    n = int(d * SR); t = np.arange(n) / SR
    L = np.zeros(n); R = np.zeros(n)
    for m in ms:
        for k, det in enumerate((-0.12, -0.05, 0.0, 0.06, 0.13)):
            f = np.full(n, mtof(m) * 2 ** (det / 12))
            s = osc_saw_blep(f) if k != 2 else np.sin(2 * np.pi * f * t)
            if k % 2:
                L += s
            else:
                R += s
    e = env_adsr(n, 0.5, 1, 1, 0.6)
    L = lp(L, bright, 2) * e; R = lp(R, bright, 2) * e
    return np.vstack([L, R]) / (len(ms) * 3)

def stab(ms, d=0.35):
    n = int(d * SR); t = np.arange(n) / SR
    x = np.zeros(n)
    for m in ms:
        for det in (-0.08, 0.08):
            x += osc_saw_blep(np.full(n, mtof(m) * 2 ** (det / 12)))
    cut = 600 + 5000 * np.exp(-t / 0.06)
    y = svf_lp(x / len(ms), cut, 0.3)
    return y * env_adsr(n, 0.004, 0.15, 0.4, 0.08)

# ------------------------------------------------------------- harmony
# 4-bar loop in D Hijaz:  D | Cm | Gm | D . Eb .
CHORDS = {
    'D': ([50, 54, 57, 62], 38), 'Cm': ([48, 51, 55, 60], 36),
    'Gm': ([50, 55, 58, 62], 43), 'Eb': ([51, 55, 58, 63], 39),
}
LOOP = [['D', 'D'], ['Cm', 'Cm'], ['Gm', 'Gm'], ['D', 'Eb']]   # half-bars

HOOK = [  # (16th step, midi, length) over 4 bars
    (0, 74, 2), (2, 72, 1), (3, 70, 1), (4, 69, 2), (6, 70, 1), (7, 69, 1), (8, 66, 2), (10, 67, 1), (11, 66, 1), (12, 63, 2), (14, 62, 2),
    (16, 67, 2), (18, 63, 1), (19, 67, 1), (20, 72, 2), (22, 70, 1), (23, 72, 1), (24, 75, 3), (27, 74, 1), (28, 72, 2), (30, 70, 1), (31, 69, 1),
    (32, 70, 2), (34, 74, 1), (35, 70, 1), (36, 67, 2), (38, 69, 1), (39, 70, 1), (40, 72, 2), (42, 70, 1), (43, 69, 1), (44, 67, 4),
    (48, 66, 1), (49, 67, 1), (50, 69, 2), (52, 70, 1), (53, 69, 1), (54, 66, 2), (56, 67, 1), (57, 66, 1), (58, 63, 2), (60, 62, 4),
]
LEAD = [
    (0, 69, 6), (6, 70, 2), (8, 69, 4), (12, 66, 2), (14, 67, 2),
    (16, 67, 4), (20, 63, 2), (22, 67, 2), (24, 72, 6), (30, 70, 2),
    (32, 74, 6), (38, 75, 2), (40, 74, 4), (44, 72, 2), (46, 70, 2),
    (48, 69, 4), (52, 70, 2), (54, 69, 2), (56, 67, 2), (58, 66, 2), (60, 63, 2), (62, 62, 2),
]

def chord_at(t, start):
    """Chord name at absolute time t for a loop starting at `start`."""
    k = int((t - start) / (BAR / 2)) % 8
    return LOOP[k // 2][k % 2]

# ----------------------------------------------------------- arrangement
def groove(t0, t1, kick_every=1, maqsum=True, clap_on=True, hats=True, riq_on=False, gain=1.0):
    t = t0
    while t < t1 - 1e-6:
        b = int(round((t - t0) / BEAT))
        beat_in_bar = b % 4
        if b % kick_every == 0:
            add('drums', kick(), t, 0.42 * gain)
        if clap_on and beat_in_bar in (1, 3):
            add('drums', clap(), t, 0.5 * gain, 0.05, rev=0.25)
        if hats:
            for s in range(4):
                acc = 1.0 if s == 2 else 0.45
                add('drums', hat(), t + s * S16, 0.26 * acc * gain, 0.35)
            if beat_in_bar in (1, 3):
                add('drums', hat(True), t + 2 * S16, 0.1 * gain, 0.35)
        if riq_on:
            for s in range(4):
                add('drums', riq(), t + s * S16, (0.16 if s % 2 else 0.08) * gain, -0.4)
        t += BEAT
    if maqsum:
        # maqsum: D T . T D . T .  (8ths)  + ka ornaments
        bt = t0
        while bt < t1 - 1e-6:
            for e8, kind, g in [(0, 'D', 1), (1, 'T', 1), (3, 'T', 0.9), (4, 'D', 0.9), (6, 'T', 1)]:
                tm = bt + e8 * BEAT / 2
                if tm < t1:
                    if kind == 'D':
                        add('drums', doum(), tm, 0.26 * g * gain, -0.15, rev=0.1)
                    else:
                        add('drums', tek(), tm, 0.36 * g * gain, 0.2, rev=0.15)
            for e16, g in [(5, 0.35), (11, 0.3), (14, 0.4), (15, 0.3)]:
                tm = bt + e16 * S16
                if tm < t1:
                    add('drums', tek(0.6), tm, 0.25 * g * gain, 0.25, rev=0.1)
            bt += BAR

def bassline(t0, t1, loop_start):
    pat = [(0, 'r', 2), (3, 'r', 1), (4, 'o', 1), (6, 'r', 1), (8, 'r', 2), (10, 'f', 1), (11, 'r', 1), (12, 'o', 1), (14, 'f', 1), (15, 'r', 1)]
    half = loop_start
    while half < t1 - 1e-6:
        bar_t = half
        for s, deg, l in pat:
            tm = bar_t + s * S16
            if tm < t0 or tm >= t1:
                continue
            ch = chord_at(tm + 1e-4, loop_start)
            r = CHORDS[ch][1]
            m = {'r': r, 'o': r + 12, 'f': r + 7}[deg]
            add('bass', bass_note(m, l * S16 * 0.95), tm, 0.42)
        half += BAR

def pads(t0, t1, loop_start, gain=0.55, bright=1600):
    t = t0
    while t < t1 - 1e-6:
        ch = chord_at(t + 1e-4, loop_start)
        d = min(BAR / 2, t1 - t)
        add('synth', pad_chord(CHORDS[ch][0], d + 0.6, bright), t, gain, rev=0.35)
        t += BAR / 2

def chord_plucks(t0, t1, loop_start):
    """Off-beat qanun chord stabs."""
    t = t0
    while t < t1 - 1e-6:
        for off in (S16 * 2,):
            tm = t + off
            ch = chord_at(tm, loop_start)
            for j, m in enumerate(CHORDS[ch][0][1:]):
                add('synth', pluck(m + 12, 0.5, 0.6, 0.99, seed=int(tm * 100) + j), tm, 0.16, -0.3 + 0.3 * j, rev=0.15)
        t += BEAT

def hook(t0, loop_start, gain=0.7, octave=0, steps=64):
    for s, m, l in HOOK:
        if s >= steps:
            continue
        tm = t0 + s * S16
        add('synth', qanun(m + octave, 0.9 + l * S16, seed=s + 11), tm, gain, 0.15, rev=0.22, dly=0.18)

# --- 0-4 s INTRO
add('fx', sub_boom(3.0, 45), 0.0, 0.55)
add('fx', shimmer(2.5, 1174.7), 0.0, 0.18, 0.3, rev=0.6)
add('synth', pad_chord([50, 57, 62], 4.3, 900), 0.0, 0.5, rev=0.5)
add('synth', oud(38, 3.5, 5), 0.0, 0.45, -0.1, rev=0.3)
tick_times = [0, 0.5, 1.0, 1.5] + [2.0 + 0.25 * k for k in range(8)]
for k, tm in enumerate(tick_times):
    add('fx', tick(1 + 0.03 * k, 0.55 + 0.03 * k), tm, 1.0, (-0.2 if k % 2 else 0.2), rev=0.3)
for s, m, l in [(0, 62, 2), (2, 63, 1), (3, 66, 1), (4, 67, 2), (6, 69, 2), (8, 70, 1), (9, 69, 1), (10, 67, 1), (11, 66, 1), (12, 63, 2)]:
    add('synth', oud(m, 1.4, s + 2), 2.0 + s * S16, 0.36, -0.15, rev=0.3, dly=0.15)
add('fx', riser(2.0, 250, 9000), 2.0, 0.22, rev=0.3)
add('fx', rev_cymbal(1.0), 3.0, 0.25)

# --- 4-12 s DROP A
add('fx', sub_boom(1.6, 50), 4.0, 0.7)
add('drums', crash(), 4.0, 0.22, 0.2, rev=0.3)
groove(4.0, 12.0)
bassline(4.0, 12.0, 4.0)
pads(4.0, 12.0, 4.0, 0.6)
chord_plucks(4.0, 12.0, 4.0)
hook(4.0, 4.0)
hook(4.0, 4.0, gain=0.22, octave=-12)
for tm in (5.8, 6.8, 7.8, 8.8):
    add('fx', whoosh(0.4, 0.8, 400, 6000), tm - 0.12, 0.3, (-0.6 if int(tm) % 2 else 0.6))
add('fx', whoosh(0.7, 0.75, 300, 7000), 11.4, 0.36)

# --- 12-20 s B SECTION (features)
groove(12.0, 18.0, kick_every=2, clap_on=False, hats=True, gain=0.85)
groove(18.0, 19.5, kick_every=1, clap_on=False, hats=True, maqsum=True, gain=0.9)
bassline(12.0, 19.75, 12.0)
pads(12.0, 20.0, 12.0, 0.7, 2200)
lead = lead_phrase(LEAD, 0)
add('synth', lead, 12.0, 1.0, -0.05, rev=0.3, dly=0.25)
for tm in (13.85, 15.85, 17.85):
    add('fx', whoosh(0.35, 0.7, 600, 4000), tm - 0.1, 0.16)
for tm in (16.55, 16.68, 16.81):   # day wheel clicks
    add('fx', tick(0.55, 0.5), tm, 0.9, 0.2, rev=0.1)
# build 18-20
add('fx', riser(2.0, 200, 10000), 18.0, 0.26, rev=0.3)
for k in range(16):
    tm = 19.0 + k * (1.0 / 16)
    add('drums', clap(), tm, 0.06 + 0.22 * (k / 15) ** 2, 0.0)
add('fx', rev_cymbal(1.0), 19.0, 0.3)

# --- 20-26 s DROP B
add('fx', sub_boom(1.8, 48), 20.0, 0.8)
add('drums', crash(), 20.0, 0.26, -0.2, rev=0.35)
for k in range(5):                      # panel slams
    add('fx', sub_boom(0.4, 70 - 6 * k), 20.0 + 0.25 * k, 0.22)
    add('drums', tek(1.0), 20.0 + 0.25 * k, 0.22, -0.5 + 0.25 * k)
groove(20.0, 26.0, riq_on=True)
bassline(20.0, 26.0, 20.0)
pads(20.0, 26.0, 20.0, 0.65, 2400)
chord_plucks(20.0, 26.0, 20.0)
hook(20.0, 20.0, gain=0.75, steps=48)
lead2 = lead_phrase([(s, m + 12, l) for s, m, l in HOOK if s < 48], 0)
add('synth', lead2, 20.0, 0.3, 0.2, rev=0.25)
for k, tm in enumerate((24.0, 24.5, 25.0, 25.5)):
    add('synth', stab(CHORDS['Gm'][0] if k < 3 else CHORDS['D'][0], 0.3), tm, 0.32, rev=0.3)
    add('fx', sub_boom(0.3, 60), tm, 0.25)

# --- 26-28 s BREAKDOWN (Arabic tagline)
add('synth', pad_chord(CHORDS['D'][0], 1.2, 900), 26.0, 0.55, rev=0.5)
add('synth', pad_chord(CHORDS['Eb'][0], 1.3, 1200), 27.0, 0.55, rev=0.5)
for s, m, l in [(0, 74, 2), (2, 72, 1), (3, 70, 1), (4, 69, 4), (8, 70, 1), (9, 69, 1), (10, 67, 1), (11, 66, 1), (12, 63, 4)]:
    add('synth', oud(m - 12, 1.5, s + 40), 26.0 + s * S16, 0.42, 0.1, rev=0.35, dly=0.2)
for k in range(4):
    add('drums', tek(0.7), 26.0 + k * BEAT + BEAT / 2, 0.12, 0.3, rev=0.4)
add('fx', whoosh(1.3, 0.6, 300, 2500), 26.05, 0.14, -0.3, rev=0.4)
add('fx', rev_cymbal(1.0), 27.0, 0.32)
add('fx', riser(1.0, 400, 8000), 27.0, 0.16)

# --- 28-33.5 END CARD
add('drums', kick(1.0), 28.0, 1.0)
add('fx', sub_boom(3.5, 42), 28.0, 0.85)
add('drums', crash(), 28.0, 0.3, 0, rev=0.5)
add('synth', stab([50, 57, 62, 66, 69], 0.6), 28.0, 0.42, rev=0.6)
add('synth', pad_chord([50, 57, 62, 66], 5.5, 1400), 28.0, 0.5, rev=0.6)
add('synth', oud(50, 4.0, 77), 28.0, 0.4, rev=0.4)
add('fx', shimmer(4.0, 1174.7), 28.45, 0.32, 0.2, rev=0.7)
for k, tm in enumerate([29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0]):
    add('fx', tick(1.0, 0.45 * (1 - k / 8)), tm, 0.9, (-0.2 if k % 2 else 0.2), rev=0.4)

# ------------------------------------------------------------------- mix
def sidechain(starts, depth=0.55, rel=0.16):
    env = np.ones(N)
    for s in starts:
        i0 = int(s * SR); n = min(int(0.5 * SR), N - i0)
        if n <= 0:
            continue
        tt_ = np.arange(n) / SR
        env[i0:i0 + n] = np.minimum(env[i0:i0 + n], 1 - depth * np.exp(-tt_ / rel))
    return lp(env, 60)

kicks = [4.0 + BEAT * k for k in range(16)] + [12.0 + BAR / 2 * k for k in range(6)] + [18.0 + BEAT * k for k in range(3)] + [20.0 + BEAT * k for k in range(12)] + [28.0]
sc = sidechain(kicks)
buses['bass'] *= sc
buses['synth'] *= (0.35 + 0.65 * sc)

# ping-pong delay (3/16)
dt = int(3 * S16 * SR)
dl = np.zeros((2, N)); fb = 0.38
src = dsend
for k in range(1, 7):
    sh = dt * k
    if sh >= N:
        break
    g = fb ** (k - 1) * 0.6
    ch = (k % 2)
    dl[ch, sh:] += src[0, :N - sh] * g * 0.5 + src[1, :N - sh] * g * 0.5
dl = np.vstack([lp(dl[0], 5000), lp(dl[1], 5000)])

# convolution reverb
ir_t = np.arange(int(2.4 * SR)) / SR
irL = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t / 0.48)
irR = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t / 0.48)
irL = lp(irL, 6000); irR = lp(irR, 6000)
irL[: int(0.012 * SR)] = 0; irR[: int(0.017 * SR)] = 0
irL /= np.sqrt((irL ** 2).sum()); irR /= np.sqrt((irR ** 2).sum())
sendf = np.vstack([hp(send[0], 250), hp(send[1], 250)])
rev = np.vstack([fftconvolve(sendf[0], irL)[:N], fftconvolve(sendf[1], irR)[:N]]) * 0.5

mix = (buses['drums'] * 0.9 + buses['bass'] * 0.85 + buses['synth'] * 0.9 + buses['fx'] * 0.85 + rev + dl * 0.6)
mix = np.vstack([hp(mix[0], 28), hp(mix[1], 28)])

# glue compressor (feed-forward, RMS)
def compress(x, thr_db=-14, ratio=2.5, att=0.01, rel=0.15):
    lvl = np.sqrt(lp((x ** 2).mean(axis=0), 30) + 1e-12)
    db = 20 * np.log10(lvl)
    over = np.maximum(db - thr_db, 0)
    gr = -over * (1 - 1 / ratio)
    g = 10 ** (gr / 20)
    return x * g

mix = compress(mix / (np.abs(mix).max() + 1e-9) * 0.9)
# soft limiter
mix = mix / np.abs(mix).max() * 1.25
mix = np.tanh(mix) / np.tanh(1.25)
# fades
fade_in = np.minimum(np.arange(N) / (0.01 * SR), 1)
fo_start = int(32.0 * SR)
fade_out = np.ones(N); fade_out[fo_start:] = np.linspace(1, 0, N - fo_start) ** 2
mix *= fade_in * fade_out
mix *= 0.89 / np.abs(mix).max()

os.makedirs('build', exist_ok=True)
pcm = (np.clip(mix.T, -1, 1) * 32767).astype('<i2')
with wave.open('build/music.wav', 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote build/music.wav', mix.shape, 'peak', np.abs(mix).max())

if os.environ.get('STEMS'):
    for k, v in list(buses.items()) + [('rev', rev), ('dly', dl)]:
        seg = v[:, int(4 * SR):int(12 * SR)].mean(axis=0)
        F = np.abs(np.fft.rfft(seg)) ** 2; f = np.fft.rfftfreq(len(seg), 1 / SR)
        lo = F[f < 120].sum() / F.sum()
        print(f'{k:6s} rms {20*np.log10(np.sqrt((seg**2).mean())+1e-9):6.1f} dB   <120Hz {100*lo:4.1f}%')
