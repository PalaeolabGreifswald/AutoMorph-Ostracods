# -*- coding: utf-8 -*-
"""
Created on Fri May  2 18:51:25 2025

@author: marle

"""

import os
import pandas as pd

# Ordnerpfad mit den CSV-Dateien
input_folder = "C:/Users/MH/Desktop/OF_new/m_RV_dors/morph2d/coordinates"
output_filename = "TiP_m_RV_coordinates.tps"
output_path = os.path.join(input_folder, output_filename)

# Liste zum Sammeln der TPS-Einträge
tps_entries = []

for filename in os.listdir(input_folder):
    if filename.endswith("_original.csv"):
        filepath = os.path.join(input_folder, filename)

        # CSV-Datei einlesen
        df = pd.read_csv(filepath, sep=",", engine="python", skipinitialspace=True)

        # Prüfen auf notwendige Spalten
        if "SampleID" in df.columns and "ObjectID" in df.columns and "x" in df.columns and "y" in df.columns:
            sample_id = df.loc[0, "SampleID"]
            object_id = df.loc[0, "ObjectID"]
            full_id = f"{sample_id}_{object_id}"

            # Nur Koordinatenspalten extrahieren
            coords = df[["x", "y"]].copy()

            # Linkester und rechtester Punkt entlang X
            leftmost_idx = coords["x"].idxmin()
            rightmost_idx = coords["x"].idxmax()

            leftmost = coords.loc[leftmost_idx]
            rightmost = coords.loc[rightmost_idx]

            #Linker Punkt, rechter Punkt, dann alle Originalpunkte
            extended_coords = pd.concat([
                pd.DataFrame([rightmost]),  # LM1
                pd.DataFrame([leftmost]), # LM2
                coords                     # Originale Landmarken
            ], ignore_index=True)

            # Sicherstellen, dass genau 102 Punkte vorhanden sind
            if len(extended_coords) != 102:
                print(f"❌ Fehler: Datei '{filename}' enthält nach Erweiterung nicht genau 102 Punkte.")
                continue

            # TPS-Block erzeugen
            tps_block = ["LM=102"]
            tps_block += [f"{row['x']} {row['y']}" for _, row in extended_coords.iterrows()]
            tps_block.append(f"ID={full_id}")
            tps_block.append("SCALE=1")

            # Zum Gesamtergebnis hinzufügen
            tps_entries.append("\n".join(tps_block))
        else:
            print(f"⚠️ Datei '{filename}' enthält nicht alle notwendigen Spalten (SampleID, ObjectID, X, Y).")

# In TPS-Datei schreiben
if tps_entries:
    with open(output_path, "w") as f:
        f.write("\n".join(tps_entries))  # Zwei Zeilen Abstand zwischen Einträgen
    print(f"✅ TPS-Datei gespeichert unter: {output_path}")
else:
    print("❌ Keine gültigen CSV-Dateien für die TPS-Erstellung gefunden.")
