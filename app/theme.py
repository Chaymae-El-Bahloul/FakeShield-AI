"""Visual identity of the FakeShield AI interface: CSS and inline SVG assets."""
import base64
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"


def h(s: str) -> str:
    """Collapse an HTML snippet to one line (avoids Markdown code-block parsing)."""
    return "".join(line.strip() for line in s.strip().splitlines())


LOGO_SVG = h("""
<svg viewBox="0 0 64 72" width="64" height="72" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="lg_logo" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#22d3ee"/><stop offset=".55" stop-color="#6366f1"/><stop offset="1" stop-color="#a855f7"/>
</linearGradient></defs>
<path d="M32 3 L58 13 V35 C58 52 46 64 32 69 C18 64 6 52 6 35 V13 Z" fill="rgba(99,102,241,.18)" stroke="url(#lg_logo)" stroke-width="4" stroke-linejoin="round"/>
<path d="M20 36 L29 45 L45 26" fill="none" stroke="url(#lg_logo)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""")

HERO_SVG = h("""
<svg class="hero-svg" viewBox="0 0 620 380" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="hs_shield" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#7dd3fc"/><stop offset=".5" stop-color="#3b82f6"/><stop offset="1" stop-color="#6d28d9"/></linearGradient>
<linearGradient id="hs_ring" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#a855f7" stop-opacity=".1"/><stop offset=".5" stop-color="#60a5fa"/><stop offset="1" stop-color="#a855f7" stop-opacity=".1"/></linearGradient>
<linearGradient id="hs_sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#7aa7ff"/><stop offset="1" stop-color="#f6c6a0"/></linearGradient>
<radialGradient id="hs_halo" cx=".5" cy=".5" r=".5"><stop offset="0" stop-color="#4f7bff" stop-opacity=".55"/><stop offset="1" stop-color="#4f7bff" stop-opacity="0"/></radialGradient>
<filter id="hs_glow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="14" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<ellipse cx="310" cy="335" rx="295" ry="42" fill="none" stroke="url(#hs_ring)" stroke-width="3"/>
<ellipse cx="310" cy="335" rx="232" ry="30" fill="none" stroke="url(#hs_ring)" stroke-width="2" opacity=".7"/>
<circle cx="320" cy="175" r="175" fill="url(#hs_halo)"/>
<g transform="translate(34 118) rotate(-7)">
<rect width="196" height="172" rx="8" fill="#d8def5"/>
<rect x="10" y="10" width="176" height="42" rx="4" fill="#1e2a5a"/>
<text x="20" y="42" font-family="Impact,Arial Black,sans-serif" font-size="32" fill="#e8ecff">NEWS</text>
<rect x="12" y="64" width="84" height="64" fill="#8fa0c9"/>
<g stroke="#7a86ad" stroke-width="4" stroke-linecap="round"><line x1="106" y1="70" x2="182" y2="70"/><line x1="106" y1="84" x2="182" y2="84"/><line x1="106" y1="98" x2="182" y2="98"/><line x1="106" y1="112" x2="170" y2="112"/><line x1="12" y1="142" x2="182" y2="142"/><line x1="12" y1="156" x2="160" y2="156"/></g>
</g>
<g transform="translate(404 104) rotate(7)">
<rect width="204" height="142" rx="8" fill="#0f1a44" stroke="#8aa4ff" stroke-width="2"/>
<rect x="6" y="6" width="192" height="130" rx="5" fill="url(#hs_sky)"/>
<polygon points="6,120 60,60 100,100 142,50 198,112 198,136 6,136" fill="#2b3b66"/>
<polygon points="6,136 40,102 82,126 122,96 198,136" fill="#1a264d"/>
</g>
<g filter="url(#hs_glow)"><path d="M310 20 L430 62 V170 C430 245 380 298 310 330 C240 298 190 245 190 170 V62 Z" fill="url(#hs_shield)" opacity=".92"/></g>
<path d="M310 20 L430 62 V170 C430 245 380 298 310 330 C240 298 190 245 190 170 V62 Z" fill="none" stroke="#bfe3ff" stroke-width="5" stroke-linejoin="round"/>
<path d="M310 46 L408 80 V170 C408 232 368 276 310 304 C252 276 212 232 212 170 V80 Z" fill="rgba(10,20,90,.45)" stroke="#8ec5ff" stroke-width="2"/>
<path d="M255 168 L295 208 L368 120" fill="none" stroke="#e8f6ff" stroke-width="22" stroke-linecap="round" stroke-linejoin="round"/>
</svg>""")


def hero_html() -> str:
    """Use app/assets/hero.(png|jpg|webp) if present, otherwise the built-in SVG."""
    for ext, mime in (("png", "png"), ("jpg", "jpeg"), ("jpeg", "jpeg"), ("webp", "webp")):
        p = ASSETS / f"hero.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img class="hero-img" src="data:image/{mime};base64,{b64}"/>'
    return HERO_SVG


def _svg(inner: str, size: int = 26) -> str:
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>')


ICONS = {
    "cpu": _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
                '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>'),
    "camera": _svg('<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>'),
    "link": _svg('<path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>'),
    "chart": _svg('<line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>'),
    "check": _svg('<polyline points="20 6 9 17 4 12"/>', 30),
    "alert": _svg('<path d="M12 8v5"/><path d="M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/>', 30),
    "tick": _svg('<polyline points="20 6 9 17 4 12"/>', 16),
    "warn": _svg('<path d="M12 8v5"/><path d="M12 17h.01"/><circle cx="12" cy="12" r="10"/>', 16),
    "cap": _svg('<path d="M22 10 12 5 2 10l10 5 10-5z"/><path d="M6 12v5c3 2 9 2 12 0v-5"/>', 18),
    "github": '<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>',
}

PALETTES = {
    "dark": {
        "bg1": "#070719", "bg2": "#130a3b", "sidebar": "#0a0a24",
        "card": "rgba(22,26,80,.55)", "border": "rgba(125,135,255,.22)",
        "text": "#f2f3ff", "muted": "#a8aee3", "input": "rgba(8,10,40,.85)",
    },
    "light": {
        "bg1": "#f5f3ff", "bg2": "#e9e6ff", "sidebar": "#ffffff",
        "card": "rgba(255,255,255,.92)", "border": "rgba(99,102,241,.28)",
        "text": "#1b1b40", "muted": "#5a5f98", "input": "#ffffff",
    },
}

_BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp{font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;color:var(--text);
background:radial-gradient(1100px 560px at 78% -8%,rgba(99,102,241,.38),transparent 60%),
radial-gradient(900px 520px at 0% 105%,rgba(168,85,247,.26),transparent 60%),
linear-gradient(160deg,var(--bg1),var(--bg2));}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"],[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{display:none !important}
.block-container{padding:1.1rem 1.8rem 2.4rem !important;max-width:1560px}
.stApp p,.stApp label,.stApp span,.stApp li{color:inherit}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,var(--sidebar),#0d0a2e);
border-right:1px solid var(--border);width:285px !important;min-width:285px !important}
section[data-testid="stSidebar"] > div{padding-top:.4rem}
.brand{text-align:center;padding:14px 0 18px}
.brand h2{margin:.35rem 0 .1rem;font-size:1.65rem;font-weight:800;letter-spacing:-.5px;color:var(--text)}
.brand h2 span{font-weight:500}
.brand p{margin:0;color:var(--muted);font-size:.92rem;line-height:1.35}
section[data-testid="stSidebar"] .stButton>button{justify-content:flex-start;width:100%;border:1px solid transparent;
background:transparent;color:var(--text);border-radius:12px;padding:.62rem .95rem;font-weight:500;box-shadow:none;gap:.65rem}
section[data-testid="stSidebar"] .stButton>button p{font-size:.98rem;font-weight:500}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(99,102,241,.16);border-color:rgba(130,140,255,.25)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"],
section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{
background:linear-gradient(90deg,rgba(99,102,241,.6),rgba(139,92,246,.35));border:1px solid rgba(170,150,255,.6);
box-shadow:0 0 20px rgba(99,102,241,.35);color:#fff}
section[data-testid="stSidebar"] .stElementContainer{margin-bottom:-.35rem}
.promo{margin-top:26px;border-radius:16px;border:1px solid var(--border);padding:20px 16px 0;text-align:center;overflow:hidden;
background:linear-gradient(180deg,rgba(18,20,70,.7),rgba(60,30,130,.45))}
.promo b{display:block;color:var(--text);font-size:1.02rem;line-height:1.4;margin-bottom:10px}
.promo svg{display:block;width:100%;height:70px}

/* ---------- top bar ---------- */
.badge{display:inline-flex;align-items:center;gap:10px;padding:9px 18px;border-radius:999px;border:1px solid var(--border);
background:var(--card);font-size:.86rem;color:var(--text);font-weight:500}
.badge i{opacity:.6;font-style:normal}
.gh{display:inline-flex;align-items:center;gap:8px;padding:8px 18px;border-radius:999px;border:1px solid rgba(190,200,255,.55);
color:var(--text) !important;text-decoration:none !important;font-weight:600;font-size:.9rem}
.gh:hover{background:rgba(99,102,241,.2)}
.avatar{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:.85rem;color:#fff;
background:linear-gradient(135deg,#6366f1,#d946ef);border:2px solid rgba(255,255,255,.35)}
div[class*="st-key-theme_btn"] button{width:42px !important;height:42px;min-height:42px;padding:0 !important;border-radius:50% !important;
background:var(--card) !important;border:1px solid var(--border) !important;color:var(--text) !important;box-shadow:none !important}

/* ---------- hero ---------- */
.hero h1{font-size:4.1rem;line-height:1.02;font-weight:800;letter-spacing:-2px;margin:.9rem 0 .6rem;color:var(--text)}
.grad{background:linear-gradient(90deg,#d946ef,#8b5cf6 55%,#6366f1);-webkit-background-clip:text;background-clip:text;color:transparent !important}
.hero h3{font-size:1.5rem;font-weight:700;margin:.2rem 0 1rem;color:var(--text)}
.hero h3 .grad{background:linear-gradient(90deg,#e879f9,#a855f7)}
.hero p{font-size:1.12rem;color:var(--muted);max-width:30rem;line-height:1.55;margin-bottom:1.1rem}
.hero-svg,.hero-img{width:100%;height:auto;display:block;max-height:400px;object-fit:contain}

/* ---------- buttons ---------- */
.block-container .stButton>button{border-radius:999px;padding:.72rem 1.5rem;font-weight:600;transition:transform .15s,box-shadow .15s}
.block-container .stButton>button[kind="primary"],.block-container .stButton>button[data-testid="stBaseButton-primary"]{
background:linear-gradient(90deg,#6366f1,#8b5cf6 55%,#a855f7);border:0;color:#fff;box-shadow:0 8px 26px rgba(124,92,255,.45)}
.block-container .stButton>button[kind="primary"]:hover,.block-container .stButton>button[data-testid="stBaseButton-primary"]:hover{
transform:translateY(-1px);box-shadow:0 12px 30px rgba(124,92,255,.6);color:#fff}
.block-container .stButton>button[kind="secondary"],.block-container .stButton>button[data-testid="stBaseButton-secondary"]{
background:transparent;border:1.5px solid rgba(190,200,255,.65);color:var(--text)}
.block-container .stButton>button[kind="secondary"]:hover,.block-container .stButton>button[data-testid="stBaseButton-secondary"]:hover{
background:rgba(99,102,241,.18);border-color:#a5b4fc;color:var(--text)}

/* ---------- feature cards ---------- */
.feat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:1.2rem 0 1.4rem}
.feat{display:flex;gap:14px;align-items:center;padding:16px 18px;border-radius:16px;background:var(--card);border:1px solid var(--border);
backdrop-filter:blur(6px)}
.feat .ic{width:54px;height:54px;border-radius:14px;display:grid;place-items:center;flex:0 0 54px}
.feat b{display:block;font-size:1.02rem;color:var(--text)}
.feat span{color:var(--muted);font-size:.88rem;line-height:1.35}
@media (max-width:1100px){.feat-grid{grid-template-columns:repeat(2,1fr)}}

/* ---------- cards (keyed containers) ---------- */
div[class*="st-key-card_"]{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:22px 24px;
backdrop-filter:blur(6px);box-shadow:0 10px 40px rgba(5,5,40,.35)}
.card-title{font-size:1.35rem;font-weight:700;margin-bottom:.8rem;color:var(--text)}
div[class*="st-key-card_res_real"]{border-color:rgba(34,197,94,.5);background:linear-gradient(180deg,rgba(6,72,54,.55),rgba(10,32,52,.55))}
div[class*="st-key-card_res_fake"]{border-color:rgba(239,68,68,.55);background:linear-gradient(180deg,rgba(100,22,44,.55),rgba(32,14,52,.55))}

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"]{background:var(--input);border:1.6px dashed rgba(120,140,255,.6);border-radius:14px;padding:26px 14px;min-height:190px;
display:flex;align-items:center;justify-content:center}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted)}
[data-testid="stFileUploaderDropzone"] button{border-radius:10px;border:1px solid var(--border);color:var(--text)}
.stTextArea label p,.stSelectbox label p{font-weight:600;color:var(--text)}
.stTextArea textarea{background:var(--input) !important;border:1px solid var(--border) !important;border-radius:12px !important;color:var(--text) !important;min-height:120px}
.stTextArea textarea::placeholder{color:var(--muted)}
div[data-baseweb="select"]>div{background:var(--input);border-color:var(--border);color:var(--text)}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--muted) !important;text-align:center}

/* ---------- result ---------- */
.res-head{display:flex;align-items:center;gap:16px;margin-bottom:.9rem}
.res-ic{width:58px;height:58px;border-radius:50%;display:grid;place-items:center;color:#fff;flex:0 0 58px}
.res-head.real .res-ic{background:#22c55e;box-shadow:0 0 0 8px rgba(34,197,94,.18)}
.res-head.fake .res-ic{background:#ef4444;box-shadow:0 0 0 8px rgba(239,68,68,.18)}
.res-title{font-size:1.5rem;font-weight:800;color:var(--text)}
.res-sub{color:var(--text);opacity:.92}
.res-sub .ok{color:#4ade80}.res-sub .bad{color:#fca5a5}
.conf{display:flex;align-items:center;gap:14px;margin:.2rem 0 1rem;color:var(--text)}
.conf .bar{flex:1;height:12px;border-radius:99px;background:rgba(255,255,255,.12);overflow:hidden}
.conf .bar div{height:100%;border-radius:99px}
.conf.real .bar div{background:linear-gradient(90deg,#22c55e,#4ade80)}
.conf.fake .bar div{background:linear-gradient(90deg,#ef4444,#fb923c)}
.res-note{font-size:.78rem;color:var(--muted);margin-top:.6rem}
.res-note code{font-size:.78rem}
.checks{list-style:none;margin:0;padding:0}
.checks li{display:flex;align-items:center;gap:10px;padding:7px 0;color:var(--text);font-size:.97rem}
.checks .dot{width:22px;height:22px;border-radius:50%;display:grid;place-items:center;flex:0 0 22px;color:#fff}
.checks li.ok .dot{background:#22c55e}.checks li.warn .dot{background:#f59e0b}
.stTabs [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--border)}
.stTabs [data-baseweb="tab"]{background:transparent;color:var(--muted);padding:.55rem 1.1rem;border-radius:10px 10px 0 0}
.stTabs [aria-selected="true"]{background:rgba(255,255,255,.08);color:var(--text)}
.stTabs [data-baseweb="tab-highlight"]{background:#8b5cf6}
.chips{display:flex;flex-wrap:wrap;gap:6px;margin:.4rem 0}
.chip{padding:4px 10px;border-radius:8px;font-size:.92rem;color:#fff;border:1px solid rgba(255,255,255,.12)}
.legend{font-size:.8rem;color:var(--muted);margin-top:.4rem}
.placeholder{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;min-height:300px;color:var(--muted)}
.placeholder svg{opacity:.55;margin-bottom:10px}

/* ---------- misc pages ---------- */
.page-title{font-size:2.1rem;font-weight:800;letter-spacing:-.8px;margin:.8rem 0 .2rem;color:var(--text)}
.page-sub{color:var(--muted);margin-bottom:1.2rem}
.info-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:1rem 0}
.info{padding:18px 20px;border-radius:16px;background:var(--card);border:1px solid var(--border)}
.info b{display:block;margin-bottom:.35rem;font-size:1.02rem;color:var(--text)}
.info span{color:var(--muted);font-size:.92rem;line-height:1.5}
.disclaimer{padding:14px 18px;border-radius:14px;border:1px solid rgba(245,158,11,.5);background:rgba(245,158,11,.1);color:var(--text);margin:1rem 0}
[data-testid="stDataFrame"]{border-radius:12px;overflow:hidden}
"""


def css(dark: bool = True) -> str:
    p = PALETTES["dark" if dark else "light"]
    root = ":root{" + ";".join(f"--{k}:{v}" for k, v in p.items()) + "}"
    return f"<style>{root}{_BASE_CSS}</style>"
