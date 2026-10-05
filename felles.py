from pathlib import Path
import numpy as np
import cv2

MAPPE = Path("facial-emotion-recognition/images")   # endre hvis mappen ligger et annet sted
UTTRYKK = ["Anger", "Contempt", "Disgust", "Fear",
           "Happy", "Neutral", "Sad", "Surprised"]
PERSONER = range(19)
FRO = 42                      # fast frø for alt som er tilfeldig

# Unntak i PCA-datasettet, funnet ved å se gjennom ansikter_oversikt.png
MIDTUTSNITT = [11]            # ansiktet fyller hele bildet, detektoren bommer
UTELATES = [(6, "Disgust"), (10, "Surprised"), (14, "Fear")]   # feil utsnitt

def finn_fil(person, uttrykk):
    for fil in (MAPPE / str(person)).iterdir():
        if fil.stem.lower() == uttrykk.lower():
            return fil
    raise FileNotFoundError(f"Fant ikke {uttrykk} for person {person}")

def les_graa(person, uttrykk, langside=512):
    """Leser ett bilde som gråtone, skalert slik at lengste side = langside.
    Returnerer verdier 0-255 (uint8)."""
    data = np.fromfile(finn_fil(person, uttrykk), dtype=np.uint8)
    bilde = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    skala = langside / max(bilde.shape)
    return cv2.resize(bilde, None, fx=skala, fy=skala,
                      interpolation=cv2.INTER_AREA)

def til_01(bilde):
    """Normaliserer fra 0-255 til [0, 1]."""
    return bilde.astype(np.float64) / 255.0

_detektor = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def ansiktsutsnitt(bilde, storrelse=64, bruk_detektor=True):
    """Klipper ut ansiktet og skalerer til storrelse x storrelse.
    Returnerer (utsnitt, funnet). Finner den ikke noe ansikt, eller
    bruk_detektor=False, brukes et kvadrat midt i bildet og funnet = False."""
    minste = min(bilde.shape) // 5
    treff = []
    if bruk_detektor:
        treff = _detektor.detectMultiScale(bilde, scaleFactor=1.1,
                                           minNeighbors=5,
                                           minSize=(minste, minste))
    if len(treff) > 0:
        x, y, b, h = max(treff, key=lambda t: t[2] * t[3])
        funnet = True
    else:
        b = h = min(bilde.shape)
        x = (bilde.shape[1] - b) // 2
        y = (bilde.shape[0] - h) // 2
        funnet = False
    utsnitt = bilde[y:y + h, x:x + b]
    utsnitt = cv2.resize(utsnitt, (storrelse, storrelse),
                         interpolation=cv2.INTER_AREA)
    return utsnitt, funnet
