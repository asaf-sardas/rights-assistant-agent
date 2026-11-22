"""Split page Documents into chunks and add them to a vector store.

Works with any LangChain VectorStore (Chroma in production, an in-memory
store in tests), so this module does not depend on a specific database.
"""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Split pages into chunks.

    Each chunk:
    - keeps its page's metadata (title, url, source) - needed for citations,
    - gets `chunk_index` and a stable `id` ("<source>#<index>"),
    - starts with the page title, so even a chunk from the middle of a page
      carries its topic (improves search).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # paragraphs first, then lines, then words - keeps related text together
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    for page in documents:
        for index, text in enumerate(splitter.split_text(page.page_content)):
            source = page.metadata["source"]
            chunks.append(
                Document(
                    id=f"{source}#{index}",
                    page_content=f"{page.metadata['title']}\n\n{text}",
                    metadata={**page.metadata, "chunk_index": index},
                )
            )
    return chunks


def index_chunks(store: VectorStore, chunks: list[Document]) -> list[str]:
    """Embed and store chunks. Returns the stored ids.

    Ids are stable, so adding the same chunk again overwrites it
    instead of creating a duplicate.
    """
    if not chunks:
        return []
    return store.add_documents(chunks, ids=[chunk.id for chunk in chunks])
