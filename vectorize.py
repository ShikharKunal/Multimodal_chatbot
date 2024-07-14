import os
import uuid
import chromadb
import numpy as np
from langchain_community.vectorstores import Chroma
from PIL import Image as _PILImage
from img_summarizer import img_summarizer
from txt_summarizer import txt_summarizer
from langchain.docstore.document import Document

print("before vectorstore")
# Create chroma
vectorstore = Chroma(
    collection_name="mm_rag", embedding_function="sentence-transformers/msmarco-distilbert-base-v3"
)
print("after vectorstore")


def vectorize_(image_output_dir_path: str, text_chunks: List[str]) -> Chroma:
    # Add image summaries

    imgs = [
        Document(
            id=[os.path.join(image_output_dir_path, img)],
            content=img_summarizer(os.path.join(image_output_dir_path, img)),
            metadata={"is_img": True}
        )
        for img in os.listdir(image_output_dir_path)
    ]
    vectorstore.add_documents(imgs)

    # Add text summaries
    texts = [
        Document(id=uuid.uuid4(), content=txt_summarizer(txt), metadata={"is_img": False}) for txt in text_chunks
    ]
    vectorstore.add_documents(texts)

    return vectorstore


