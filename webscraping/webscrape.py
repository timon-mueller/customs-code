import requests
from bs4 import BeautifulSoup
import json
import csv
import re

def scrape_zolltarif_chapters():
    url = "https://www.tariffnumber.com"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    print(f"Rufe Daten von {url} ab...")
    response = requests.get(url, headers=headers)
    response.encoding = 'utf-8' 
    
    if response.status_code != 200:
        print(f"Fehler beim Abruf der Seite. Status Code: {response.status_code}")
        return
        
    soup = BeautifulSoup(response.text, 'html.parser')
    chapters_data = []
    
    # 1. Wir suchen alle übergeordneten Container für die Kapitel-Zeilen
    rowgroups = soup.find_all('div', class_='rowgroup')
    
    for group in rowgroups:
        # 2. Suche den Link im linken Bereich (col-sm-2)
        link_tag = group.find('a')
        # 3. Suche das Beschreibungs-Feld im rechten Bereich (col-sm-10)
        desc_tag = group.find('div', class_='col-sm-10')
        
        if link_tag and desc_tag:
            link_text = link_tag.get_text(strip=True) # Gibt z.B. "Chapter 01"
            
            # Sicherstellen, dass es sich wirklich um ein Kapitel handelt (Case-insensitive check)
            if "chapter" in link_text.lower():
                # Extrahiere die reine Nummer (entfernt das Wort "Chapter" case-insensitive)
                chapter_number = re.sub(r'(?i)chapter', '', link_text).strip()
                # Extrahiere den Text der Beschreibung
                description = desc_tag.get_text(strip=True)
                
                chapters_data.append({
                    "chapter": chapter_number,
                    "description": description
                })

    print(f"\nErfolgreich {len(chapters_data)} Kapitel extrahiert.")

    # Als JSON speichern
    json_filename = "zolltarif_chapters.json"
    with open(json_filename, 'w', encoding='utf-8') as json_file:
        json.dump(chapters_data, json_file, ensure_ascii=False, indent=4)
    print(f"Daten als JSON gespeichert: {json_filename}")

    # Als CSV speichern (mit Semikolon getrennt für Excel/Calc)
    csv_filename = "zolltarif_chapters.csv"
    with open(csv_filename, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["chapter", "description"], delimiter=';')
        writer.writeheader()
        writer.writerows(chapters_data)
    print(f"Daten als CSV gespeichert: {csv_filename}")

if __name__ == "__main__":
    scrape_zolltarif_chapters()