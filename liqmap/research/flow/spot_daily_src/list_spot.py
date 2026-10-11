import re, sys, urllib.request, urllib.parse
out = sys.argv[1]
base = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
marker = ""
syms = []
while True:
    q = urllib.parse.urlencode({"delimiter": "/", "prefix": "data/spot/monthly/klines/", "marker": marker})
    x = urllib.request.urlopen(f"{base}?{q}", timeout=60).read().decode()
    ps = re.findall(r"<Prefix>data/spot/monthly/klines/([^<]+)/</Prefix>", x)
    syms += ps
    if "<IsTruncated>true</IsTruncated>" not in x:
        break
    m = re.search(r"<NextMarker>([^<]+)</NextMarker>", x)
    marker = m.group(1) if m else f"data/spot/monthly/klines/{ps[-1]}/"
open(out, "w").write("\n".join(syms))
print(len(syms))
