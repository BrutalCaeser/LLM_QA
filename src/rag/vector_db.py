
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from langchain_core.documents import Document
import hashlib
import logging

class VectorDB:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True,
                           "batch_size": 64 } # Optimized for RTX 2060 memory  # Important for FAISS
        )
        self.expected_dim = 384  # bge-small-en-v1.5 dime

        self.logger = logging.getLogger(__name__)
    def _format_context(self, row: pd.Series) -> str:
        """Convert DataFrame row to natural language context"""
        return (
            f"Hotel: {row['hotel']}, Country: {row['country']}, "
            f"Arrival: {row['arrival_date'].strftime('%Y')}, "
            f"Lead Time: {row['lead_time']} days, Nights: {row['total_nights']}, "
            f"ADR: €{row['adr']:.2f}, Cancelled: {'Yes' if row['is_canceled'] else 'No'}, "
            #f"Market Segment: {row['market_segment']}"
        )

    def _create_documents(self, df: pd.DataFrame) -> list[Document]:
        """Create documents with hotel-specific metadata"""
        docs = []
        duplicate_hashes = set()

        for _, row in df.iterrows():
            content = self._format_context(row)
            content_hash = hashlib.sha256(content.encode()).hexdigest()  # Safer than md5

            if content_hash in duplicate_hashes:
                continue

            duplicate_hashes.add(content_hash)

            docs.append(Document(
                page_content=content,
                metadata={
                    "country": row["country"],
                    "cancelled": bool(row["is_canceled"]),
                    "lead_time": int(row["lead_time"]),
                    "adr": float(row["adr"]),
                    "arrival_year_month": row["arrival_date"].strftime("%Y"),
                    #"market_segment": row["market_segment"],
                    "content_hash": content_hash
                }
            ))

        return docs
    def create_vector_store(self):
        """Create and save FAISS index"""
        df = pd.read_parquet(self.project_root / "data" / "processed" / "cleaned_bookings.parquet")
        if df.empty:
            raise ValueError("Empty dataset loaded")
        docs = self._create_documents(df)

        self.logger.info(f"Creating FAISS index with {len(docs)} bookings")
        #df["context"] = df.apply(self._format_context, axis=1)

        #loader = DataFrameLoader(df, page_content_column="context")
        #docs = loader.load()

        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local(str(self.project_root / "data" / "faiss_index"))
        return vector_store

    def get_vector_store(self):
        """Load existing or create new FAISS index"""
        index_path = self.project_root / "data" / "faiss_index"
        if not index_path.exists():
            print("FAISS index not found. Creating a new one...")
            return self.create_vector_store()
        try:
            return FAISS.load_local(
                str(index_path),
                self.embeddings,
                allow_dangerous_deserialization=True)  # Trust your own index
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Rebuilding...")
            return self.create_vector_store()


#vector_db = VectorDB()
#vector_db.create_vector_store()
#vector_store = vector_db.get_vector_store()
#print(vector_store.similarity_search("Bookings from Germany"))

# Test with booking data patterns
"""vector_db = VectorDB()
store = vector_db.get_vector_store()

# Find similar luxury cancellations
results = store.similarity_search(
    query="luxury hotel cancellations",
    k=5,
)

# After your similarity search code
print(f"Found {len(results)} relevant bookings:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(f"Content: {doc.page_content}")
    print("Metadata:")
    print(f"  - Country: {doc.metadata['country']}")
    print(f"  - Cancelled: {doc.metadata['cancelled']}")
    print(f"  - ADR: €{doc.metadata['adr']:.2f}")
    print(f"  - Lead Time: {doc.metadata['lead_time']} days")
    print(f"  - Market Segment: {doc.metadata['market_segment']}")
    print("-" * 80)"""