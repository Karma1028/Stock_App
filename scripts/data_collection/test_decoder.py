import asyncio
from googlenewsdecoder import new_decoderv1

url = "https://news.google.com/rss/articles/CBMiigFBVV95cUxQMEhEMXhkckZhOHJNN3o2SVZxbUFrejVPN2hQSFRjZy1vUEpqTFlBZ09HUnNTclB1ZTdhblhTc3I0Vm85bWUyS3VrLTdzdGlsa1pNNkhlRTQzZmtaMnNaMmhCWVNZQ3l5eXhQWDFMZndYdHFmS2p4Y3VpXzJuTndLdzlKemJGNkZtMmc?oc=5"

def test_decoder():
    try:
        decoded_url = new_decoderv1(url)
        print("Decoded success:", decoded_url)
    except Exception as e:
        print("Error decoding:", e)

if __name__ == "__main__":
    test_decoder()
