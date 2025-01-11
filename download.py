import requests

# URL do pliku PDF
url = "http://api.sejm.gov.pl/eli/acts/DU/2020/20/text.pdf"

# Wysyłanie zapytania GET, aby pobrać plik
response = requests.get(url)

# Sprawdzanie, czy zapytanie zakończyło się sukcesem
if response.status_code == 200:
    # Zapisanie pliku PDF do lokalnego pliku
    with open("akt_prawny_2020_20.pdf", "wb") as f:
        f.write(response.content)
    print("Plik został pobrany i zapisany jako 'akt_prawny_2020_20.pdf'.")
else:
    print(f"Error: {response.status_code}")
