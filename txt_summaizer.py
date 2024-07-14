from transformers import pipeline
print("before summarizer")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print("after summarizer")
def txt_summarizer(text: str) -> str:
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
