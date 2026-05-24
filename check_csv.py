import pandas as pd

df = pd.read_csv("data/tmdb_5000_movies.csv")
print("Columns:", list(df.columns))
print()
print("First row sample:")
print(df.iloc[0][["title", "id"]].to_string())
print()
# Check if any image-related columns exist
img_cols = [c for c in df.columns if any(x in c.lower() for x in ["poster", "image", "photo", "backdrop", "img"])]
print("Image-related columns:", img_cols)
print()
print("Sample id values:", df["id"].head(5).tolist())
