"""
Démonstration visuelle : sauvegarde toutes les étapes du pipeline en images.
"""
import matplotlib
matplotlib.use('Agg')  # backend non-interactif pour produire des fichiers

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as pil

from empreintes import binarisation, etape3, minutie, Reel_T


# --- 1. Image originale ---
img_orig = np.array(pil.open('Belle_empreinte.jpg'))
plt.figure(figsize=(5, 6))
plt.imshow(img_orig)
plt.title("1) Image originale")
plt.axis('off')
plt.savefig('etape1_originale.png', bbox_inches='tight', dpi=110)
plt.close()

# --- 2. Binarisation ---
tab_bin = binarisation('Belle_empreinte.jpg')
plt.figure(figsize=(5, 6))
plt.imshow(tab_bin, cmap='binary')
plt.title("2) Binarisation")
plt.axis('off')
plt.savefig('etape2_binarisee.png', bbox_inches='tight', dpi=110)
plt.close()

# --- 3. Squelettisation ---
sq = etape3(tab_bin)
plt.figure(figsize=(5, 6))
plt.imshow(sq, cmap='binary')
plt.title("3) Squelettisation (Zhang-Suen)")
plt.axis('off')
plt.savefig('etape3_squelette.png', bbox_inches='tight', dpi=110)
plt.close()

# --- 4. Minuties brutes ---
T, D = minutie(sq)
plt.figure(figsize=(5, 6))
plt.imshow(sq, cmap='binary')
if T:
    plt.scatter([p[1] for p in T], [p[0] for p in T], color='blue', s=18, label=f'Terminaisons ({len(T)})')
if D:
    plt.scatter([p[1] for p in D], [p[0] for p in D], color='red', s=18, label=f'Divergences ({len(D)})')
plt.legend(loc='lower right', fontsize=8)
plt.title("4) Minuties brutes")
plt.axis('off')
plt.savefig('etape4_minuties_brutes.png', bbox_inches='tight', dpi=110)
plt.close()

# --- 5. Minuties filtrees par ellipse ---
Tf = Reel_T(T)
Df = Reel_T(D)
plt.figure(figsize=(5, 6))
plt.imshow(sq, cmap='binary')
if Tf:
    plt.scatter([p[1] for p in Tf], [p[0] for p in Tf], color='blue', s=22, label=f'Terminaisons ({len(Tf)})')
if Df:
    plt.scatter([p[1] for p in Df], [p[0] for p in Df], color='red', s=22, label=f'Divergences ({len(Df)})')
plt.legend(loc='lower right', fontsize=8)
plt.title("5) Minuties filtrees par ellipse centrale")
plt.axis('off')
plt.savefig('etape5_minuties_filtrees.png', bbox_inches='tight', dpi=110)
plt.close()

print(f"OK - terminaisons retenues : {len(Tf)} | divergences retenues : {len(Df)}")
