import sys
sys.path.insert(0, '.')
from app.db.vector_store import get_vector_store

vs = get_vector_store()
results = vs._collection.get(include=["metadatas"], limit=3)
for m in results["metadatas"]:
    print("title:      ", m.get("title", "N/A"))
    print("poster_url: ", m.get("poster_url", "MISSING"))
    print("poster_path:", m.get("poster_path", "MISSING"))
    print()
