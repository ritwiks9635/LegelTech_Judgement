from typing import Dict, Any, Optional, List
from pymongo import MongoClient, errors
import re


# ============================================================
# 1. Connect to MongoDB Atlas (Colab Safe)
# ============================================================
def get_mongo_client(
    uri: str,
    db_name: str = "legaltech_poc"
):
    """
    MongoDB Atlas connector for Colab testing.
    """

    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()   
        return client[db_name]

    except errors.ServerSelectionTimeoutError:
        raise RuntimeError(
            "❌ MongoDB connection failed.\n"
            "✔ Check username/password\n"
            "✔ Check IP whitelist (0.0.0.0/0)\n"
            "✔ Check cluster is running"
        )
    except Exception as e:
        raise RuntimeError(f"MongoDB error: {str(e)}")


# ============================================================
# 2. Save Parsed Judgment Metadata
# ============================================================
def save_metadata(
    db,
    collection: str,
    metadata: Dict[str, Any],
    unique_key: str = "title"
):
    if unique_key not in metadata:
        raise ValueError(f"Missing unique key: {unique_key}")

    db[collection].update_one(
        {unique_key: metadata[unique_key]},
        {"$set": metadata},
        upsert=True
    )
    return True


# ============================================================
# 3. Load by Case Title
# ============================================================
def load_metadata(
    db,
    collection: str,
    title: str
) -> Optional[Dict[str, Any]]:

    return db[collection].find_one(
        {"title": title},
        {"_id": 0}
    )


# ============================================================
# 4. List All Stored Cases
# ============================================================
def list_all_cases(db, collection: str) -> List[str]:

    cursor = db[collection].find({}, {"title": 1, "_id": 0})
    return [doc["title"] for doc in cursor]


# ============================================================
# 5. Simple Keyword Search (TESTING PURPOSE)
# ============================================================
def search_cases(
    db,
    collection: str,
    query: str,
    limit: int = 5
):
    """
    MongoDB keyword search to verify DB works.
    Searches in:
    - facts
    - holding
    - issues
    """

    regex = re.compile(query, re.IGNORECASE)

    cursor = db[collection].find(
        {
            "$or": [
                {"facts": regex},
                {"holding": regex},
                {"issues": regex}
            ]
        },
        {
            "_id": 0,
            "title": 1,
            "court": 1,
            "date": 1,
            "holding": 1
        }
    ).limit(limit)

    return list(cursor)
