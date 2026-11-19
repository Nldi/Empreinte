import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
import math
from copy import deepcopy
import sqlite3


# -------------------------
# 1) Fonction de binarisation
# -------------------------

def binarisation(image_path, seuil_rgb=125, seuil_gris=100, taille_max=400):

    img = pil.open(image_path)

    w, h = img.size
    if max(w, h) > taille_max:
        if w > h:
            new_w = taille_max
            new_h = int(h * taille_max / w)
        else:
            new_h = taille_max
            new_w = int(w * taille_max / h)
        img = img.resize((new_w, new_h), pil.LANCZOS)

    tab = np.array(img)

    # Image couleur RGB
    if len(tab.shape) == 3:

        h, l, r = tab.shape
        tab_bin = np.zeros((h, l), dtype=np.uint8)

        for i in range(h):
            for j in range(l):

                c = (
                    int(tab[i][j][0]) +
                    int(tab[i][j][1]) +
                    int(tab[i][j][2])
                ) // 3

                if c < seuil_rgb:
                    tab_bin[i][j] = 1
                else:
                    tab_bin[i][j] = 0

    # Image niveaux de gris
    else:

        h, l = tab.shape[:2]
        tab_bin = np.zeros((h, l), dtype=np.uint8)

        for i in range(h):
            for j in range(l):

                c = tab[i][j]

                if c < seuil_gris:
                    tab_bin[i][j] = 1
                else:
                    tab_bin[i][j] = 0

    return tab_bin


# -------------------------
# 2) Premiere squelettisation : convolution
# -------------------------

def convolution(tab, N):

    h, l = tab.shape
    tab_output = np.zeros((h, l), dtype=np.uint8)

    for i in range(1, h-1):
        for j in range(1, l-1):

            pixel = 0

            for k in range(3):
                for m in range(3):

                    pixel = pixel + N[k][m] * tab[i+k-1][j+m-1]

            if pixel < 0:
                pixel = 0

            elif pixel >= 2:
                pixel = 1

            tab_output[i][j] = pixel

    return tab_output


# -------------------------
# 3) Fonctions utilitaires
# -------------------------

def indice(a, b, A):

    if a < 0 or a >= A.shape[0] or b < 0 or b >= A.shape[1]:
        return 0

    return int(A[a, b])


def pt_noir(i, j, A):
    return indice(i, j, A) == 1


def Bp(i, j, A):

    s = (
        indice(i-1, j-1, A) +
        indice(i-1, j, A) +
        indice(i-1, j+1, A) +
        indice(i, j-1, A) +
        indice(i, j+1, A) +
        indice(i+1, j-1, A) +
        indice(i+1, j, A) +
        indice(i+1, j+1, A)
    )

    return 2 <= s <= 6


def Ap(i, j, A):

    L = [
        indice(i-1, j, A),
        indice(i-1, j+1, A),
        indice(i, j+1, A),
        indice(i+1, j+1, A),
        indice(i+1, j, A),
        indice(i+1, j-1, A),
        indice(i, j-1, A),
        indice(i-1, j-1, A)
    ]

    n = 0

    for k in range(len(L)):

        if L[k] == 0 and L[(k+1) % len(L)] == 1:
            n += 1

    return n == 1


def P024(i, j, A):

    return (
        indice(i-1, j, A) *
        indice(i, j+1, A) *
        indice(i+1, j, A)
    ) == 0


def P246(i, j, A):

    return (
        indice(i, j+1, A) *
        indice(i+1, j, A) *
        indice(i, j-1, A)
    ) == 0


def P026(i, j, A):

    return (
        indice(i-1, j, A) *
        indice(i, j+1, A) *
        indice(i, j-1, A)
    ) == 0


def P046(i, j, A):

    return (
        indice(i-1, j, A) *
        indice(i+1, j, A) *
        indice(i, j-1, A)
    ) == 0


# -------------------------
# 4) Etapes 1 et 2 de Zhang-Suen
# -------------------------

def etape1(A):

    to_remove = []
    h, w = A.shape

    for i in range(1, h-1):
        for j in range(1, w-1):

            if (
                pt_noir(i, j, A)
                and Bp(i, j, A)
                and Ap(i, j, A)
                and P024(i, j, A)
                and P246(i, j, A)
            ):

                to_remove.append((i, j))

    for (i, j) in to_remove:
        A[i, j] = 0

    return A


def etape2(A):

    to_remove = []
    h, w = A.shape

    for i in range(1, h-1):
        for j in range(1, w-1):

            if (
                pt_noir(i, j, A)
                and Bp(i, j, A)
                and Ap(i, j, A)
                and P026(i, j, A)
                and P046(i, j, A)
            ):

                to_remove.append((i, j))

    for (i, j) in to_remove:
        A[i, j] = 0

    return A


# -------------------------
# 5) Boucle Zhang-Suen (etape 3)
# -------------------------

def etape3(A, max_iter=1000):

    A = A.copy()

    prev = np.zeros_like(A)

    it = 0

    while not np.array_equal(A, prev) and it < max_iter:

        prev = A.copy()

        A = etape1(A)
        A = etape2(A)

        it += 1

    return A


# -------------------------
# 6) Minuties : CN
# -------------------------

def CN_at(i, j, A):

    neigh = [
        indice(i-1, j, A),
        indice(i-1, j+1, A),
        indice(i, j+1, A),
        indice(i+1, j+1, A),
        indice(i+1, j, A),
        indice(i+1, j-1, A),
        indice(i, j-1, A),
        indice(i-1, j-1, A)
    ]

    s = 0

    for k in range(len(neigh)):
        s += abs(neigh[k] - neigh[(k+1) % len(neigh)])

    return s // 2


def minutie(A):

    terminaisons = []
    bifurcations = []

    u, v = A.shape

    for i in range(1, u-1):
        for j in range(1, v-1):

            if A[i, j] == 1:

                s = CN_at(i, j, A)

                if s == 1:
                    terminaisons.append([i, j])

                elif s == 3:
                    bifurcations.append([i, j])

    return terminaisons, bifurcations


# -------------------------
# 7) Filtrage par ellipse + affichage
# -------------------------

def Ellipse(L, T):

    if len(T) == 0:
        return False

    ys = [p[0] for p in T]
    xs = [p[1] for p in T]

    cy = (min(ys) + max(ys)) / 2
    cx = (min(xs) + max(xs)) / 2

    a = (max(ys) - min(ys)) / 2
    b = (max(xs) - min(xs)) / 2

    if a == 0 or b == 0:
        return False

    a *= 0.8
    b *= 0.8

    if (
        ((L[0] - cy) ** 2) / (a ** 2) +
        ((L[1] - cx) ** 2) / (b ** 2)
    ) <= 1:

        return True

    return False


def Reel_T(T):

    return [p for p in T if Ellipse(p, T)]


def point(T, D, M, p):

    plt.figure(figsize=(6, 8))
    plt.axis('off')
    plt.imshow(M, cmap='binary')

    if len(T) > 0:
        plt.scatter([pt[1] for pt in T], [pt[0] for pt in T], color='blue', s=p)

    if len(D) > 0:
        plt.scatter([pt[1] for pt in D], [pt[0] for pt in D], color='red', s=p)

    plt.show()


# -------------------------
# 8) Transformations : translation, rotation, distance
# -------------------------

def rotation(T, centre, theta):

    N = []

    for i in range(len(T)):

        x, y = T[i][0], T[i][1]

        Nx = math.floor(
            (x - centre[0]) * np.cos(theta)
            - (y - centre[1]) * np.sin(theta)
            + centre[0]
        )

        Ny = math.floor(
            (x - centre[0]) * np.sin(theta)
            + (y - centre[1]) * np.cos(theta)
            + centre[1]
        )

        N.append([Nx, Ny])

    return N


def translation(M, v):

    M1 = deepcopy(M)

    for i in range(len(M)):
        M1[i][0] += v[0]
        M1[i][1] += v[1]

    return M1


def distance(P, Q):

    return math.sqrt((
        (P[0] - Q[0]) ** 2 +
        (P[1] - Q[1]) ** 2
    ))


# -------------------------
# 9) Comparaison de deux jeux de minuties
# -------------------------

def _apparier_1_a_1(H, E, seuil_dist):
    """
    Fonction utilitaire : compte combien de points de H peuvent etre
    apparies a un point distinct de E (chaque point de E ne peut servir
    qu'une fois) avec distance <= seuil_dist.

    Renvoie le nombre d'appariements trouves.
    """
    E_utilises = [False] * len(E)
    z = 0
    for l in range(len(H)):
        meilleur_i = -1
        meilleure_dist = seuil_dist + 1
        for i in range(len(E)):
            if E_utilises[i]:
                continue
            d = distance(E[i], H[l])
            if d <= seuil_dist and d < meilleure_dist:
                meilleure_dist = d
                meilleur_i = i
        if meilleur_i != -1:
            E_utilises[meilleur_i] = True
            z += 1
    return z


def transfoT(T, E, n, seuil_dist=20, ratio_min=0.45):
    """
    Cherche une translation v de norme <= n telle que T translate de v
    correspond a E.
    """

    if len(T) == 0 or len(E) == 0:
        return False

    # Une proportion ratio_min des minuties de la plus grande des deux
    # empreintes doit etre appariee, avec un minimum de 8.
    seuil_match = max(8, int(ratio_min * max(len(T), len(E))))

    for j in range(-n, n+1):
        for k in range(-n, n+1):

            v = [j, k]
            H = translation(T, v)

            z = _apparier_1_a_1(H, E, seuil_dist)

            if z >= seuil_match:
                return True

    return False


def transfoR(T, E, n, seuil_dist=20, ratio_min=0.45):
    """
    Cherche une rotation de T (autour du centre de sa boite englobante,
    d'angle 2*pi*i/n pour i = 0..n-1) qui rapproche T de E.
    """

    if len(T) == 0 or len(E) == 0:
        return False

    xs = [p[0] for p in T]
    ys = [p[1] for p in T]
    c = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2]

    theta = (2 * np.pi) / n
    seuil_match = max(8, int(ratio_min * max(len(T), len(E))))

    for i in range(n):

        rT = rotation(T, c, i * theta)

        z = _apparier_1_a_1(rT, E, seuil_dist)

        if z >= seuil_match:
            return True

    return False


# -------------------------
# 10) Base de donnees SQLite
# -------------------------

def reset():

    connection = sqlite3.connect('B_Empreinte.db')
    B = connection.cursor()

    B.execute('''DROP TABLE IF EXISTS coord''')

    table = """
    CREATE TABLE IF NOT EXISTS coord(
        type VARCHAR(1),
        idempreinte INT,
        y INT,
        x INT
    );
    """

    B.execute(table)

    connection.commit()
    connection.close()


def ajout(T, D, id):

    connection = sqlite3.connect('B_Empreinte.db')
    B = connection.cursor()

    for type_minutie, L in (('T', T), ('D', D)):
        for p in L:
            B.execute(
                "INSERT INTO coord VALUES(?,?,?,?)",
                (type_minutie, int(id), int(p[0]), int(p[1]))
            )

    connection.commit()
    connection.close()


def parcourir():

    connection = sqlite3.connect('B_Empreinte.db')
    B = connection.cursor()

    B.execute("SELECT type, idempreinte, y, x FROM coord")

    for rep in B.fetchall():
        print(rep)

    connection.close()


def comparaison(T, n):
    """
    Verifie si l'empreinte T appartient a la base, en la comparant
    aux empreintes 0..n-1. Renvoie True des qu'une correspondance est trouvee.
    """

    connection = sqlite3.connect('B_Empreinte.db')
    B = connection.cursor()

    for rep in range(n):

        B.execute(
            "SELECT y, x FROM coord WHERE idempreinte = ? AND type = 'T'",
            (rep,)
        )

        t = [[i, j] for i, j in B.fetchall()]

        if len(t) == 0:
            continue

        if transfoT(T, t, 10) or transfoR(T, t, 10):
            connection.close()
            return True

    connection.close()

    return False


def liste_empreintes():
    """Renvoie la liste des id d'empreintes presentes dans la base."""

    connection = sqlite3.connect('B_Empreinte.db')
    B = connection.cursor()

    B.execute("SELECT DISTINCT idempreinte FROM coord ORDER BY idempreinte")

    ids = [row[0] for row in B.fetchall()]

    connection.close()
    return ids


# -------------------------
# 11) Identification : trouver QUELLE empreinte correspond
# -------------------------

def identifier(T):
    """
    Parcourt TOUTES les empreintes de la base
    et renvoie l'ID de celle qui correspond (ou None si aucune).

    Renvoie (id_trouve, liste_de_tous_les_matchs).
    """

    ids = liste_empreintes()
    correspondances = []

    print(f"Recherche parmi {len(ids)} empreintes dans la base...")

    for rep in ids:

        # On recupere les terminaisons de l'empreinte rep
        connection = sqlite3.connect('B_Empreinte.db')
        B = connection.cursor()

        B.execute(
            "SELECT y, x FROM coord WHERE idempreinte = ? AND type = 'T'",
            (rep,)
        )

        t = [[i, j] for i, j in B.fetchall()]
        connection.close()

        if len(t) == 0:
            continue

        # On essaie translation puis rotation
        match_T = transfoT(T, t, 10)
        match_R = False
        if not match_T:
            match_R = transfoR(T, t, 10)

        if match_T or match_R:
            type_match = "translation" if match_T else "rotation"
            print(f"  -> empreinte {rep} : MATCH ({type_match})")
            correspondances.append(rep)
        else:
            print(f"  -> empreinte {rep} : pas de correspondance")

    if len(correspondances) == 0:
        return None, []

    if len(correspondances) > 1:
        print(f"ATTENTION : {len(correspondances)} correspondances trouvees")

    return correspondances[0], correspondances


# -------------------------
# 12) Pipeline complet image -> minuties
# -------------------------

def traiter_empreinte(image_path):
    """
    Enchaine binarisation -> squelettisation -> extraction des minuties
    -> filtrage par ellipse. Renvoie (terminaisons, divergences).
    """

    tab = binarisation(image_path)
    sq = etape3(tab)
    T, D = minutie(sq)
    T = Reel_T(T)
    D = Reel_T(D)

    return T, D


# =====================================================================
# EXEMPLE D'UTILISATION : enrolement de plusieurs empreintes,
# puis identification d'une nouvelle empreinte
# =====================================================================

if __name__ == '__main__':

    print("=" * 60)
    print("Etape 1 : extraction des minuties pour les 4 empreintes")
    print("=" * 60)

    print("\nTraitement de Belle_empreinte.jpg (doigt A)...")
    T_belle, D_belle = traiter_empreinte('Belle_empreinte.jpg')
    print(f"  terminaisons : {len(T_belle)}   divergences : {len(D_belle)}")

    print("\nTraitement de empreinte-digitale-1.jpg (doigt B)...")
    T_ed1, D_ed1 = traiter_empreinte('empreinte-digitale-1.jpg')
    print(f"  terminaisons : {len(T_ed1)}   divergences : {len(D_ed1)}")

    print("\nTraitement de empreinte5_3.jpg (doigt C)...")
    T_e53, D_e53 = traiter_empreinte('empreinte5_3.jpg')
    print(f"  terminaisons : {len(T_e53)}   divergences : {len(D_e53)}")

    print("\nTraitement de capasse.jpg (doigt A apres rotation)...")
    T_capasse, D_capasse = traiter_empreinte('capasse.jpg')
    print(f"  terminaisons : {len(T_capasse)}   divergences : {len(D_capasse)}")

    # --- Enrolement des 3 doigts distincts en base ----
    print("\n" + "=" * 60)
    print("Etape 2 : enrolement dans la base SQLite")
    print("=" * 60)

    reset()
    ajout(T_belle, D_belle, 0)
    ajout(T_ed1,   D_ed1,   1)
    ajout(T_e53,   D_e53,   2)

    print(f"\nBase remplie : ids = {liste_empreintes()}")
    print("  id 0 = Belle_empreinte (doigt A)")
    print("  id 1 = empreinte-digitale-1 (doigt B)")
    print("  id 2 = empreinte5_3 (doigt C)")

    # --- Identification de capasse ----
    print("\n" + "=" * 60)
    print("Etape 3 : identification de 'capasse'")
    print("=" * 60)
    print("\n>>> capasse = Belle_empreinte tournee de ~270 degres")
    print(">>> Le programme doit dire : id 0")
    print()

    print(f"capasse appartient a la base : {comparaison(T_capasse, 3)}\n")

    id_trouve, tous = identifier(T_capasse)

    print(f"\nRESULTAT : id identifie = {id_trouve}")
    print(f"           liste des matches : {tous}")

    if id_trouve == 0 and tous == [0]:
        print("           OK : capasse correctement reconnue comme Belle_empreinte")
    elif id_trouve == 0 and len(tous) > 1:
        print(f"           PARTIEL : Belle reconnue, mais aussi confondue avec {tous}")
    elif id_trouve is None:
        print("           FAUX NEGATIF : aucune correspondance trouvee")
    else:
        print(f"           FAUX POSITIF : capasse confondue avec id {id_trouve}")

    # --- Comparaisons croisees ----
    print("\n" + "=" * 60)
    print("Etape 4 : comparaisons croisees")
    print("=" * 60)
    print("Aucune paire de doigts differents ne doit matcher.")
    print("La paire Belle/capasse (meme doigt) doit matcher.\n")

    paires = [
        ('Belle',   T_belle,   'ed1',     T_ed1,    False),
        ('Belle',   T_belle,   'e53',     T_e53,    False),
        ('ed1',     T_ed1,     'e53',     T_e53,    False),
        ('Belle',   T_belle,   'capasse', T_capasse, True),
        ('ed1',     T_ed1,     'capasse', T_capasse, False),
        ('e53',     T_e53,     'capasse', T_capasse, False),
    ]

    print(f"{'paire':<25} {'trans':<8} {'rot':<8} {'verdict'}")
    print("-" * 55)

    faux_pos = 0
    faux_neg = 0
    for nom1, T1, nom2, T2, meme_doigt in paires:
        mt = transfoT(T1, T2, 10)
        mr = transfoR(T1, T2, 10) if not mt else False
        match = mt or mr

        if meme_doigt:
            verdict = "OK (meme doigt)" if match else "FAUX NEGATIF"
            if not match:
                faux_neg += 1
        else:
            verdict = "FAUX POSITIF" if match else "OK"
            if match:
                faux_pos += 1

        paire_str = f"{nom1} vs {nom2}"
        print(f"{paire_str:<25} {('MATCH' if mt else 'no'):<8} {('MATCH' if mr else 'no'):<8} {verdict}")

    print(f"\n>>> Faux positifs : {faux_pos} / 5 paires differentes")
    print(f">>> Faux negatifs : {faux_neg} / 1 paire identique")