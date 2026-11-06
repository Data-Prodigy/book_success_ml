"""
Genre buckets from Open Library subject tags.

Open Library tags a work with free-text subjects ("Thrillers", "Detective and mystery stories",
"Love stories", ...). We map them onto a few broad buckets. The genre is something the publisher
knows before launch, so it is a legitimate predictor. (The NUMBER of tags is deliberately not used:
popular books accumulate tags over time, which would leak the outcome.)
"""
import re

BUCKETS = ["thriller", "mystery_crime", "romance", "fantasy", "scifi", "horror",
           "historical", "literary", "young_adult", "other"]
GENRE_COLUMNS = [f"genre_{b}" for b in BUCKETS]

# Priority = order of this dict when two buckets tie on matching tags.
_KEYWORDS = {
    "thriller": ["thriller", "suspense", "espionage", "spy ", "spies", "conspirac"],
    "mystery_crime": ["mystery", "detective", "crime", "murder", "whodunit", "police procedural", "private investigators"],
    "romance": ["romance", "love stories", "contemporary women"],
    "fantasy": ["fantasy", "magic", "wizard", "dragon", "fairy tale"],
    "scifi": ["science fiction", "dystopia", "time travel", "extraterrestrial", "space opera", "cyberpunk"],
    "horror": ["horror", "ghost", "supernatural", "vampire", "occult", "haunted", "monsters"],
    "historical": ["historical", "world war", "civil war", "regency", "medieval"],
    "literary": ["literary", "classic", "psychological fiction", "african fiction", "english fiction", "american fiction"],
    "young_adult": ["young adult", "juvenile", "children", "teen"],
}


def genre_from_subjects(subjects) -> str:
    """'Thrillers | Suspense | Fiction' -> 'thriller'. No match / no tags -> 'other'."""
    tags = [t.strip().lower() for t in re.split(r"\|", str(subjects)) if t.strip()] if isinstance(subjects, str) else []
    best, best_n = "other", 0
    for bucket, words in _KEYWORDS.items():
        n = sum(any(w in tag for w in words) for tag in tags)
        if n > best_n:       # strictly greater: ties go to the earlier (higher-priority) bucket
            best, best_n = bucket, n
    return best


def clean_genre(value) -> str:
    """Normalise a user-supplied genre ('Mystery/Crime', 'sci-fi') onto a bucket; unknown -> 'other'."""
    v = re.sub(r"[^a-z]+", "_", str(value).strip().lower()).strip("_")
    aliases = {"sci_fi": "scifi", "science_fiction": "scifi", "mystery": "mystery_crime", "crime": "mystery_crime",
               "ya": "young_adult", "historical_fiction": "historical", "thrillers": "thriller"}
    v = aliases.get(v, v)
    return v if v in BUCKETS else "other"


def add_genre_columns(df):
    """One-hot genre_* columns (all always present, in BUCKETS order) from df['genre']."""
    out = df.copy()
    g = out["genre"].map(clean_genre)
    for b in BUCKETS:
        out[f"genre_{b}"] = (g == b).astype("int64")
    return out


def fill_unknown_with_author_genre(df, author_col: str = "author_key"):
    """A book whose tags match no bucket ('other') takes its author's most common NAMED genre.

    Why: sparse tags mostly mean an obscure book, so leaving them as "other" turned the genre
    column into a popularity proxy (77% of "other" books were Misses). Filling with the author's
    usual genre removes that signal. Authors with no named genre at all stay "other".
    """
    out = df.copy()
    named = out[out["genre"] != "other"]
    usual = named.groupby(author_col)["genre"].agg(lambda g: g.mode().iat[0])   # ties: alphabetical, deterministic
    fill = out[author_col].map(usual)
    mask = (out["genre"] == "other") & fill.notna()
    out.loc[mask, "genre"] = fill[mask]
    return out